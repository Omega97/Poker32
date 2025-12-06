# scripts/pipelines/pipeline_gto.py
import random
from pathlib import Path
from src.agents.agent_grok import AgentEquilibrium
from src.training import training
from src.utils import inspect_policy


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------

    _MODEL_NAME = Path(__file__).stem.split("_")[-1]
    _POLICY_PATH = Path(f"../../models/{_MODEL_NAME}.json")
    _AGENT_CLASS = AgentEquilibrium

    # ---------- safety check ----------
    if _POLICY_PATH.exists():
        ans = input(f"{_POLICY_PATH.name} exists — continue?")


    rng = random.Random(42)
    config = {
        "learning_rate": 0.08,
        "temperature": 1.0,
        "logit_range": 25.0,
        "momentum": 0.92,
        "entropy_bonus": 0.012,   # ← the magic
        "n_epochs": 10_000,
        "n_cycles": 100,          # → 10M hands base
    }

    # ---------------- TRAINING PIPELINE ----------------
    print(f"Training {_MODEL_NAME.upper()} — the final equilibrium seeker")

    # Phase 1: Exploration
    training(_AGENT_CLASS, _POLICY_PATH, config, rng)

    # Phase 2–5: Progressive refinement
    for damping in [0.88, 0.94, 0.97, 0.995]:
        config["damping"] = damping
        config["learning_rate"] *= 0.8
        training(_AGENT_CLASS, _POLICY_PATH, config, rng)

    print("\nEQUILIBRIUM TRAINING COMPLETE")
    inspect_policy(_POLICY_PATH, show_proba=True)
