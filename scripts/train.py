import random
from pathlib import Path
from src.utils import inspect_policy
from src.training import training


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------

    # Model info
    _MODEL_NAME = "new"
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.json")
    # _POLICY_PATH = Path(f"..\\models\\tournament_2025\\{_MODEL_NAME}.pkl")

    # Config
    _RNG = random.Random(42)
    _CONFIG = {"learning_rate": 0.1,
               "temperature": 1.0,
               "init_range": 0.1,
               "logit_range": 20,
               "momentum": 0.9,  # on accumulated rewards and counts
               "damping": 0.5,   # on logits
               "n_epochs": 5_000,
               "n_cycles": 50}

    # ---------------- TRAINING PIPELINE ----------------

    training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # --------------- INSPECT THE RESULTS ---------------

    inspect_policy(file_path=_POLICY_PATH, show_proba=True)
