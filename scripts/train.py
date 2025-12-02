from pathlib import Path
from src.rl_agent import AgentRL
from src.poker32 import Poker32
from src.utils import inspect_policy


def training(file_path, config):

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


if __name__ == '__main__':

    # Model info
    _MODEL_NAME = "new"
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.pkl")

    # These are the realistic numbers for convergence
    # Try gradually lowering the 'min_logit'.
    # The 'learning_rate' should be lowered at the end.
    _CONFIG = {
        "learning_rate": 0.1,
        "temperature": 4.0,
        "min_logit": -10.,
        "momentum": 0.9,
        "n_epochs": 1_000,
        "n_cycles": 100,
    }

    # Run training and inspect policy
    training(file_path=_POLICY_PATH, config=_CONFIG)
    inspect_policy(file_path=_POLICY_PATH, show_proba=True)
