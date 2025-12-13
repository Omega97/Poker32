from pathlib import Path
import random
from src.agents.rl_agent import AgentRL, load_rl_agent
from src.poker32 import Poker32


def test_1(rng=random.Random(0)):
    _MODEL_NAME = "agentrl_v2"
    _AGENT_CLASS = AgentRL
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.json")
    agent = load_rl_agent(_POLICY_PATH, rng=rng, training=True, verbose=True)

    print(f"> {len(agent.logits)} decision points loaded.")

    rewards = [(1, -1), (2, -2), (1, -1), (2, -2), (1, -1),
               (-4, 4), (-2, 2), (-2, 2), (-4, 4), (2, -2),
               (-2, 2), (2, -2)]

    # Init the game
    game = Poker32(rng)

    # Init players
    players = (agent, agent)

    # Play the game
    for i, reward in enumerate(rewards):
        print(f'\n===== Game {i+1} =====')
        results = game.play(players)
        print('\nResults:')
        n = max(map(len, results))
        for k, v in results.items():
            print(f"{k:>{n}} {v}")
        assert results["rewards"] == reward, f'{results["rewards"]}, {reward}'


if __name__ == '__main__':
    test_1()
