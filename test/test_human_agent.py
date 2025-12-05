import random
from src.poker32 import Poker32
from src.agent import HumanAgent
from src.rl_agent import load_rl_agent


def main(agent_path, log_path, seed:int | None = 42):
    rng = random.Random(seed)

    # Init agents (human and bot)
    human = HumanAgent(name="Hero", log_path=log_path)
    bot = load_rl_agent(agent_path, rng=rng, training=False)
    players = (bot, human)
    print('\nPlayers:')
    for player in players:
        print(f"- {player.name}")

    # Init game instance
    game = Poker32(rng=rng)

    # Match loop
    while True:

        # Play a hand
        game.play(players)

        # To break the loop
        command = input('').lower().strip()
        if command in {'q', 'quit', 'stop', 'break'}:
            break

    human.plot_cumulative_returns()


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------
    _AGENT_NAME = "v4"
    _AGENT_PATH = f"..\\models\\{_AGENT_NAME}.json"
    _LOG_PATH = "..\\data\\hero.json"
    _SEED = None
    # _SEED = 42
    # ---------------------------------------------------

    main(agent_path=_AGENT_PATH,
         log_path=_LOG_PATH,
         seed=_SEED)
