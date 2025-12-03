from src.poker32 import Poker32
from src.agent import HumanAgent
from src.rl_agent import load_rl_agent


_AGENT_PATH = "..\\models\\w2_64M.pkl"


if __name__ == '__main__':

    # Init agents (human and bot)
    human = HumanAgent(name="Hero")
    bot = load_rl_agent(_AGENT_PATH, training=False)
    players = [human, bot]

    # Init game instance
    game = Poker32()

    # Match loop
    while True:

        # Play a hand
        game.play(players)

        # To break the loop
        command = input('').lower().strip()
        if command in {'q', 'quit', 'stop', 'break'}:
            break

        # Rotate positions
        players = [players[-1]] + players[:-1]
