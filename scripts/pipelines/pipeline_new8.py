import random
from pathlib import Path
import pathlib
from src.utils import inspect_policy
from src.training import training
from src.agents.rl_agent import AgentRL


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------

    _MODEL_NAME = pathlib.Path(__file__).stem.split("_")[-1]
    _POLICY_PATH = Path(f"..\\..\\models\\{_MODEL_NAME}.json")
    _AGENT_CLASS = AgentRL

    # ---------- safety check ----------
    if _POLICY_PATH.exists() and _POLICY_PATH.stat().st_size:
        ans = input(f"{_POLICY_PATH.name} already exists – overwrite?")
    else:
        print(f'Training "{_MODEL_NAME}" ({_POLICY_PATH})')

    # Config
    _RNG = random.Random(42)
    _CONFIG = {"learning_rate": 0.05,  # step length for each spot
               "temperature": 1.0,  # modifier for the policy sampling
               "init_range": 0.1,  # initial range for the logits
               "logit_range": 20,  # logits are capped between +/- this value
               "momentum": 0.8,  # decay on accumulated rewards and counts
               "damping": 0.5,  # on logits
               "n_epochs": 10_000,  # number of hands per cycle
               "n_cycles": 50}  # number of updates

    # ---------------- TRAINING PIPELINE ----------------

    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    _CONFIG["damping"] = 0.9
    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    _CONFIG["damping"] = 0.95
    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    _CONFIG["damping"] = 0.98
    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    _CONFIG["damping"] = 0.99
    training(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)

    # --------------- INSPECT THE RESULTS ---------------

    inspect_policy(file_path=_POLICY_PATH, show_proba=True)
