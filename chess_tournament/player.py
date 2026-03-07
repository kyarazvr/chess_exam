from __future__ import annotations

import random
from typing import Optional, List, Tuple

import chess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from chess_tournament.players import Player
except Exception:
    class Player:
        def __init__(self, name: str = "Player"):
            self.name = name

        def get_move(self, fen: str):
            raise NotImplementedError


class TransformerPlayer(Player):
    """
    Fast tactical player with light transformer tie-break.

    Strategy:
    - Generate all legal moves
    - Immediately play checkmate if available
    - Filter out moves that allow mate in one if possible
    - Rank moves with fast chess heuristics + shallow one-ply safety
    - Use the transformer only to break ties among top few candidates
    """

    def __init__(
        self,
        name: str = "TransformerPlayer",
        model_name: str = "gpt2",
        seed: int = 0,
        search_depth: int = 1,
        candidate_pool_size: int = 8,
        lm_top_k: int = 4,
        lm_weight: float = 0.05,
        avoid_mate_in_one: bool = True,
    ):
        super().__init__(name)
        self.rng = random.Random(seed)

        self.search_depth = max(0, int(search_depth))
        self.candidate_pool_size = int(candidate_pool_size)
        self.lm_top_k = int(lm_top_k)
        self.lm_weight = float(lm_weight)
        self.avoid_mate_in_one = bool(avoid_mate_in_one)

        # General decoder model (allowed by assignment)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,
            chess.BISHOP: 3.3,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }

        self.center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}

    # -------------------------
    # Transformer helpers
    # -------------------------
    def _prompt(self, fen: str) -> str:
        return (
            "You are a strong chess player.\n"
            f"FEN: {fen}\n"
            "Choose the best move in UCI.\n"
            "Best move:"
        )

    @torch.no_grad()
    def _logprob_suffix(self, prompt: str, suffix: str) -> float:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        full_ids = self.tokenizer(prompt + suffix, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model(full_ids)
        logits = outputs.logits

        p_len = prompt_ids.shape[1]
        f_len = full_ids.shape[1]
        if f_len <= p_len:
            return float("-inf")

        suffix_token_ids = full_ids[0, p_len:f_len]
        total_lp = 0.0

        for j, tok_id in enumerate(suffix_token_ids):
            pos = p_len + j
            pred_logits = logits[0, pos - 1, :]
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            total_lp += float(log_probs[int(tok_id)].item())

        return total_lp

    # -------------------------
    # Chess evaluation helpers
    # -------------------------
    def _material_score(self, board: chess.Board, pov: chess.Color) -> float:
        us = 0.0
        them = 0.0
        for piece_type, value in self.piece_values.items():
            us += len(board.pieces(piece_type, pov)) * value
            them += len(board.pieces(piece_type, not pov)) * value
        return us - them

    def _center_bonus(self, move: chess.Move) -> float:
        return 0.25 if move.to_square in self.center_squares else 0.0

    def _development_bonus(self, board: chess.Board, move: chess.Move) -> float:
        piece = board.piece_at(move.from_square)
        if piece is None:
            return 0.0

        score = 0.0
        if board.fullmove_number <= 10:
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                score += 0.30
            if piece.piece_type == chess.QUEEN:
                score -= 0.50
        return score

    def _castling_bonus(self, board: chess.Board, move: chess.Move) -> float:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.KING:
            return 0.0

        if abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 2:
            return 1.20
        return 0.0

    def _queen_hangs_immediately(self, board: chess.Board, move: chess.Move, pov: chess.Color) -> bool:
        b = board.copy(stack=False)
        b.push(move)

        queen_squares = list(b.pieces(chess.QUEEN, pov))
        if not queen_squares:
            return False

        queen_squares = set(queen_squares)
        for opp in b.legal_moves:
            if b.is_capture(opp) and opp.to_square in queen_squares:
                return True
        return False

    def _hanging_penalty(self, board_after_move: chess.Board, pov: chess.Color) -> float:
        """
        Penalize pieces that are immediately capturable after our move.
        """
        penalty = 0.0

        for opp in board_after_move.legal_moves:
            if not board_after_move.is_capture(opp):
                continue

            captured = board_after_move.piece_at(opp.to_square)
            if captured is not None and captured.color == pov:
                penalty -= 0.18 * self.piece_values.get(captured.piece_type, 0.0)

        return penalty

    def _static_eval(self, board: chess.Board, pov: chess.Color) -> float:
        if board.is_checkmate():
            return -10000.0 if board.turn == pov else 10000.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = self._material_score(board, pov)

        mobility = board.legal_moves.count()
        score += 0.03 * (mobility if board.turn == pov else -mobility)

        if board.is_check():
            score += 0.20 if board.turn != pov else -0.20

        return score

    def _one_ply_score(self, board: chess.Board, move: chess.Move, pov: chess.Color) -> float:
        """
        Our move, then assume opponent picks their strongest reply by static eval.
        """
        b = board.copy(stack=False)
        b.push(move)

        if b.is_checkmate():
            return 10000.0

        replies = list(b.legal_moves)
        if not replies:
            return self._static_eval(b, pov)

        best_reply_for_opp = float("-inf")
        for reply in replies:
            b.push(reply)
            val = self._static_eval(b, not pov)
            b.pop()
            if val > best_reply_for_opp:
                best_reply_for_opp = val

        return -best_reply_for_opp

    def _move_heuristic(self, board: chess.Board, move: chess.Move, pov: chess.Color) -> float:
        score = 0.0

        mover = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square) if board.is_capture(move) else None

        if board.is_capture(move):
            if captured is not None:
                score += 0.60 + 1.15 * self.piece_values.get(captured.piece_type, 0.0)
            if mover is not None and captured is not None:
                score -= 0.12 * self.piece_values.get(mover.piece_type, 0.0)

        if move.promotion is not None:
            score += 3.0

        score += self._center_bonus(move)
        score += self._development_bonus(board, move)
        score += self._castling_bonus(board, move)

        b = board.copy(stack=False)
        b.push(move)

        if b.is_checkmate():
            return 20000.0

        if b.is_check():
            score += 1.20

        if self._queen_hangs_immediately(board, move, pov):
            score -= 2.50

        score += self._hanging_penalty(b, pov)
        score += 0.20 * self._static_eval(b, pov)

        return score

    def _allows_opponent_mate_in_one(self, board: chess.Board, move: chess.Move) -> bool:
        b = board.copy(stack=False)
        b.push(move)

        for opp in b.legal_moves:
            b.push(opp)
            is_mate = b.is_checkmate()
            b.pop()
            if is_mate:
                return True
        return False

    def _book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Tiny opening book to avoid silly starts.
        """
        if len(board.move_stack) == 0 and board.turn == chess.WHITE:
            for uci in ("e2e4", "d2d4", "c2c4"):
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    return mv

        if len(board.move_stack) == 1 and board.turn == chess.BLACK:
            for uci in ("e7e5", "c7c5", "d7d5"):
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    return mv

        return None

    def _priority_moves(self, board: chess.Board, legal: List[chess.Move], pov: chess.Color) -> List[chess.Move]:
        mates = []
        checks = []
        captures = []
        promotions = []
        castling = []
        others = []

        for m in legal:
            b = board.copy(stack=False)
            b.push(m)

            if b.is_checkmate():
                mates.append(m)
                continue

            if m.promotion is not None:
                promotions.append(m)
                continue

            if board.is_capture(m):
                captures.append(m)
                continue

            if b.is_check():
                checks.append(m)
                continue

            piece = board.piece_at(m.from_square)
            if piece is not None and piece.piece_type == chess.KING:
                if abs(chess.square_file(m.from_square) - chess.square_file(m.to_square)) == 2:
                    castling.append(m)
                    continue

            others.append(m)

        ordered = mates + checks + captures + promotions + castling

        if len(ordered) >= self.candidate_pool_size:
            return ordered[: self.candidate_pool_size]

        others = sorted(
            others,
            key=lambda m: self._move_heuristic(board, m, pov),
            reverse=True,
        )
        ordered.extend(others[: self.candidate_pool_size - len(ordered)])
        return ordered

    # -------------------------
    # Required interface
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)

        if board.is_game_over():
            return None

        legal = list(board.legal_moves)
        if not legal:
            return None

        my_color = board.turn

        # 1. Tiny opening book
        book = self._book_move(board)
        if book is not None:
            return book.uci()

        # 2. Immediate mate if available
        for m in legal:
            b = board.copy(stack=False)
            b.push(m)
            if b.is_checkmate():
                return m.uci()

        # 3. Avoid allowing mate in one if possible
        safe_legal = legal
        if self.avoid_mate_in_one:
            filtered = [m for m in legal if not self._allows_opponent_mate_in_one(board, m)]
            if filtered:
                safe_legal = filtered

        # 4. Build a small strong candidate pool
        candidates = self._priority_moves(board, safe_legal, my_color)

        # 5. Score with heuristics + shallow search
        scored: List[Tuple[chess.Move, float]] = []
        for m in candidates:
            h = self._move_heuristic(board, m, my_color)
            if self.search_depth <= 0:
                b = board.copy(stack=False)
                b.push(m)
                p1 = self._static_eval(b, my_color)
            else:
                # Depth 1 search (opponent best reply)
                p1 = self._one_ply_score(board, m, my_color)
            total = 1.8 * h + 0.80 * p1
            scored.append((m, total))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 6. Transformer tie-break only among top 3-4
        if self.lm_weight > 0.0 and len(scored) > 1:
            prompt = self._prompt(fen)
            top_k = min(max(1, self.lm_top_k), 4, len(scored))
            rescored = []

            for i, (m, base) in enumerate(scored):
                if i < top_k:
                    lp = self._logprob_suffix(prompt, " " + m.uci())
                    rescored.append((m, base + self.lm_weight * lp))
                else:
                    rescored.append((m, base))

            rescored.sort(key=lambda x: x[1], reverse=True)
            scored = rescored

        # 7. Deterministic best move, with slight random tie-break
        if len(scored) >= 2 and abs(scored[0][1] - scored[1][1]) < 0.10:
            return self.rng.choice(scored[:2])[0].uci()

        return scored[0][0].uci()
