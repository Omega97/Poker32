from pathlib import Path
import random
from src.agents.rl_agent import AgentRL, load_rl_agent
from src.poker32 import Poker32


_TEST_INFOSETS = {
        ("2", "RRRR"): "c",
        ("2", "RRD"): "c",
        ("2", "RDR"): "c",
        ("2", "RT"): "c",
        ("2", "DD"): "c",
        ("2", "DRR"): "c",
        ("2", "TR"): "c",
        ("2", "Q"): "c",
        ("2", "cRRRR"): "c",
        ("2", "cRRD"): "c",
        ("2", "cRDR"): "c",
        ("2", "cDRR"): "c",
        ("2", "cDD"): "c",
        ("2", "cTR"): "c",
        ("2", "cQ"): "c",
    }


def test_1(rng=random.Random(0)):
    _MODEL_NAME = "agentrl_v2"
    _AGENT_CLASS = AgentRL
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.json")
    agent = load_rl_agent(_POLICY_PATH,
                          rng=rng,
                          training=True,
                          verbose=True)

    print(f"> {len(agent.logits)} decision points loaded.")

    rewards = [(1, -1), (2, -2), (1, -1), (2, -2), (1, -1),
               (-4, 4), (-2, 2), (-2, 2), (-4, 4), (2, -2),
               (-2, 2), (2, -2)]
    # rewards = rewards[:2]

    # Init the game
    game = Poker32(rng)

    # Init players
    players = (agent, agent)

    # Play the game
    for i, reward in enumerate(rewards):
        print(f'\n===== Game {i + 1} =====')
        print(f'> Playing game...')
        results = game.play(players)

        # Show and verify results
        print('\nResults:')
        n = max(map(len, results))
        for k, v in results.items():
            print(f"{k:>{n}} {v}")
        assert results["rewards"] == reward, f'{results["rewards"]}, {reward}'


def test_2():
    _MODEL_NAME = "agentrl_v2"
    _AGENT_CLASS = AgentRL
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.json")
    agent = load_rl_agent(_POLICY_PATH,
                          training=True,
                          verbose=True)

    print(f"> {len(agent.logits)} decision points loaded.")

    # Init the game
    game = Poker32()

    # Init players
    players = (agent, agent)

    # Play the game
    i = 0
    while True:
        i += 1
        print(f'\n===== Playing game {i} =====')
        results = game.play(players, button=i % 2, hole_cards=("A", "2"))

        # Show and verify results
        print('\nResults:')
        n = max(map(len, results))
        for k, v in results.items():
            print(f"{k:>{n}} {v}")

        if ("2", results["branch"][:-1]) in _TEST_INFOSETS:
            input()


if __name__ == '__main__':
    # test_1()
    test_2()
