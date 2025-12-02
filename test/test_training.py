import pickle
import math
from pathlib import Path
from src.rl_agent import AgentRL
from src.poker32 import Poker32, GAME_MOVES


_MODEL_NAME = "new"
_POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.pkl")


def test_training(file_path=_POLICY_PATH):

    # These are the realistic numbers for convergence
    config = {
        "learning_rate": 0.05,
        "temperature": 1.0,
        "min_logit": -10,
        "momentum": 0.9,
        "n_epochs": 20_000,
        "n_cycles": 300,
    }

    # Load the agent with training mode enabled
    if file_path.exists():
        print(f"Loading existing policy from {file_path}")
        agent = AgentRL.load(str(file_path), config=config, training=True)
        print(f"Resuming from {agent.games_played:,} games")
    else:
        print("No existing policy found → starting from scratch")
        agent = AgentRL(config=config, training=True)

    # Game instance
    game = Poker32()
    print("Starting Poker32 RL training (additive logit, T=1.0)")
    print(f"Target: {config['n_epochs'] * config['n_cycles']:,} games\n")

    # Training loop
    for cycle in range(config["n_cycles"]):
        for _ in range(config["n_epochs"]):
            game.play((agent, agent), verbose=False)

        print(f"\rCycle {cycle+1}/{config['n_cycles']} | "
              f"Games played: {agent.games_played:,}", end='')

    agent.save(file_path)
    print(f"\nTraining complete! Policy saved.")


def test_inspect_policy(path: str = _POLICY_PATH, show_proba: bool = True):
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Trained for {data['games_played']:,} games")
    print(f"Config: {data['config']}")
    print(f"Total infosets learned: {len(data['logits'])}")  # 416
    print()

    if show_proba:
        print("Softmax probabilities (min-normalized logits):")
    else:
        print("Max-normalized logits (best action = 0.00):")

    # Sort for consistent output
    for (hole, branch), action_dict in sorted(data['logits'].items()):
        if not action_dict:
            continue  # safety

        s = f'"{branch}"'
        print(f'{hole} {s:7}', end=' →  ')

        logits = list(action_dict.values())
        actions = list(action_dict.keys())

        if show_proba:
            # Min-normalization + softmax
            min_l = min(logits)
            exps = [math.exp(l - min_l) for l in logits]
            total = sum(exps)
            probs = [e / total for e in exps]

            for act, p in zip(actions, probs):
                print(f"{act} ={p:6.1%}", end="  ")
        else:
            # Max-normalization: best action = 0.0
            max_l = max(logits)
            for act, l in zip(actions, logits):
                normalized = l - max_l
                print(f"{act} ={normalized:5.1f}", end="  ")

        print()  # newline
    print("\nInspection complete.")


if __name__ == '__main__':
    # test_training()
    test_inspect_policy()
    # test_inspect_policy(show_proba=False)
