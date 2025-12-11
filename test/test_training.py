from pathlib import Path
from src.agents.rl_agent import AgentRL
from src.poker32 import Poker32
from src.utils import inspect_policy


_MODEL_NAME = "new"
_POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.pkl")


def test_training(file_path=_POLICY_PATH):

    # These are the realistic numbers for convergence
    config = {
        "learning_rate": 0.05,
        "temperature": 1.0,
        "logit_range": 10,
        "momentum": 0.9,
        "batch_size": 20_000,
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
    print(f"Target: {config['batch_size'] * config['n_cycles']:,} games\n")

    # Training loop
    for cycle in range(config["n_cycles"]):
        for _ in range(config["batch_size"]):
            game.play((agent, agent))

        print(f"\rCycle {cycle+1}/{config['n_cycles']} | "
              f"Games played: {agent.games_played:,}", end='')

    agent.save(file_path)
    print(f"\nTraining complete! Policy saved.")


def test_inspect_policy(path: str = _POLICY_PATH, show_proba: bool = True):
    inspect_policy(path, show_proba)


if __name__ == '__main__':
    # test_training()
    test_inspect_policy()
    # test_inspect_policy(show_proba=False)
