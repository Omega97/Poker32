import random
from src.agent import RandomAgent
from src.poker32 import Poker32


def test_random_agent(rng=random.Random(3), verbose=True):

    # Init the game
    game = Poker32(rng)

    # Init players
    players = (RandomAgent(rng, name="Agent_1", verbose=verbose),
               RandomAgent(rng, name="Agent_2", verbose=verbose))

    # Play the game
    for i in range(4):
        print()
        results = game.play(players)
        if verbose:
            print('\nResults:')
            for k, v in results.items():
                print(f"{k:>12} {v}")


if __name__ == '__main__':
    test_random_agent()
