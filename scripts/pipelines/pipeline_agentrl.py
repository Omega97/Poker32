"""
Train AgentUCB with a long schedule and progressive damping.
"""
import random
from pathlib import Path
from src.utils import inspect_policy
from src.training import training
from src.agents.rl_agent import AgentRL


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------
    _MODEL_NAME = "agentrl"
    _POLICY_PATH = Path(f"..\\..\\models\\{_MODEL_NAME}.json")
    _AGENT_CLASS = AgentRL

    _RNG = random.Random(42)

    _CONFIG = {
        "learning_rate": 0.1,
        "temperature": 1.0,
        "init_range": 0.1,
        "logit_range": 20,
        "momentum": 0.9,
        "damping": 0.9,
        "n_epochs": 10_000,
        "n_cycles": 200
    }

    # ---------- safety check ----------
    if _POLICY_PATH.exists():
        input(f"Warning: {_POLICY_PATH.name} already exists. Continue?")

    # ---------------- TRAINING PIPELINE ----------------

    # Phase 1: exploration
    training(_AGENT_CLASS, _POLICY_PATH, _CONFIG, _RNG)

    # Phase 2: more stable refinement
    _CONFIG["damping"] = 0.99
    _CONFIG["momentum"] = 0.99
    training(_AGENT_CLASS, _POLICY_PATH, _CONFIG, _RNG)

    # ------------------- INSPECTION --------------------

    print("\n=== TRAINING COMPLETE ===\n")
    inspect_policy(_POLICY_PATH, show_proba=True)
