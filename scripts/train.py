from pathlib import Path
import random
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
               "min_logit": -10,
               "momentum": 0.9,
               "n_epochs": 10_000,
               "n_cycles": 100}

    # ------------------- TRAINING ---------------------

    _CONFIG["learning_rate"] = 0.1
    _CONFIG["n_cycles"] = 100
    _CONFIG["n_epochs"] = 20_000
    _CONFIG["min_logit"] = -4
    training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    _CONFIG["learning_rate"] = 0.05
    _CONFIG["n_cycles"] = 100
    _CONFIG["n_epochs"] = 20_000
    _CONFIG["min_logit"] = -10
    training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # _CONFIG["n_epochs"] = 10_000
    # _CONFIG["n_cycles"] = 200
    # _CONFIG["min_logit"] = -4
    # training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    #
    # _CONFIG["min_logit"] = -10
    # training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # _CONFIG["n_epochs"] = 10_000
    # _CONFIG["min_logit"] = -2
    # training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    #
    # _CONFIG["min_logit"] = -3
    # training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    #
    # _CONFIG["min_logit"] = -5
    # training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    #
    # _CONFIG["n_epochs"] = 20_000
    # _CONFIG["learning_rate"] = 0.05
    # _CONFIG["min_logit"] = -10
    # training(file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # Inspect the results
    inspect_policy(file_path=_POLICY_PATH, show_proba=False)
