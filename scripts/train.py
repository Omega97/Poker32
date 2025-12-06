import random
from pathlib import Path
from src.utils import inspect_policy
from src.training import training
from src.agents.rl_agent import AgentRL


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------

    # Model info
    _MODEL_NAME = "_test"
    _AGENT_CLASS = AgentRL
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.json")
    # _POLICY_PATH = Path(f"..\\models\\tournament_2025\\{_MODEL_NAME}.json")

    # Config
    _RNG = random.Random(42)
    _CONFIG = {"learning_rate": 0.05,
               "temperature": 1.0,
               "init_range": 0.1,
               "logit_range": 20,
               "momentum": 0.99,  # on accumulated rewards and counts
               "damping": 1.,   # on logits
               "n_epochs": 10_000,
               "n_cycles": 50}

    # ---------------- TRAINING PIPELINE ----------------

    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # --------------- INSPECT THE RESULTS ---------------

    inspect_policy(file_path=_POLICY_PATH, show_proba=True)
