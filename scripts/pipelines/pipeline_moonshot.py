import random
from pathlib import Path
from src.training import training
from src.utils import inspect_policy
from src.agents.agent_moonshot import AgentCRM as AgentMoonshot


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------
    _MODEL_NAME = "moonshot"
    _POLICY_PATH = Path(f"..\\..\\models\\{_MODEL_NAME}.json")
    _AGENT_CLASS = AgentMoonshot

    _RNG = random.Random(42)
    _CONFIG = {"learning_rate": 0.05,  # step length for each spot
               "init_range": 0.1,  # initial range for the logits
               "exploration": 0.1,  # Initial epsilon
               "min_exploration": 0.05,  # Floor for epsilon
               "decay": 0.999,  # Epsilon decay per hand
               "on_policy_cap": 0.20,
               "use_cfr_plus": True,
               "logit_range": 20,  # logits are capped between +/- this value
               "n_epochs": 10_000,  # number of hands per cycle
               "n_cycles": 300}  # number of updates

    # ---------- safety check ----------
    if _POLICY_PATH.exists() and _POLICY_PATH.stat().st_size:
        ans = input(f"{_POLICY_PATH.name} already exists – continue?")
    else:
        print(f'Training "{_MODEL_NAME}" ({_POLICY_PATH})')

    # ---------------- TRAINING PIPELINE ----------------
    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    # _CONFIG['learning_rate'] /= 10
    # training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # --------------- INSPECT THE RESULTS ---------------

    inspect_policy(file_path=_POLICY_PATH, show_proba=True)
