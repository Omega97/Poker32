from src.poker32 import Poker32
from src.agent import HumanAgent
from src.rl_agent import load_rl_agent


_AGENT_PATH = "..\\models\\v4.pkl"


if __name__ == '__main__':

    # Play against your best bot
    human = HumanAgent(name="Hero")
    bot = load_rl_agent(_AGENT_PATH, training=False)

    # Match loop
    while True:
        game = Poker32()
        game.play((human, bot))   # human acts first
        game.play((bot, human))   # bot acts first

        # Break the loop
        command = input().lower().strip()
        if command in {'q', 'quit', 'stop'}:
            break
