import random
from src.agent import RandomAgent
from src.poker32 import Poker32


def test_random_agent(rng=random.Random(3), verbose=True):

    # Init the game
    game = Poker32(rng)
    if verbose:
        print(game.hole_cards)

    # Init players
    players = (RandomAgent(rng, verbose=verbose),
               RandomAgent(rng, verbose=verbose))

    # Play the game
    results = game.play(players, verbose=verbose)
    if verbose:
        print(results)


if __name__ == '__main__':
    test_random_agent()
