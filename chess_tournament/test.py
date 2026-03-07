try:
    from chess_tournament import Game, RandomPlayer
    from chess_tournament.player import TransformerPlayer
except ModuleNotFoundError:
    # Allow running this file directly from within the package folder.
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from chess_tournament import Game, RandomPlayer
    from chess_tournament.player import TransformerPlayer

wins = 0
losses = 0
draws = 0

for i in range(200):
    if i % 2 == 0:
        game = Game(TransformerPlayer(), RandomPlayer("Random"), max_half_moves=300)
        result, scores, illegal = game.play()

        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1
    else:
        game = Game(RandomPlayer("Random"), TransformerPlayer(), max_half_moves=300)
        result, scores, illegal = game.play()

        if result == "0-1":
            wins += 1
        elif result == "1-0":
            losses += 1
        else:
            draws += 1

    print(f"Game {i+1}: {result}")

print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
