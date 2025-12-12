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
    _CONFIG = {
        # --- CFR core -------------------------------------------------------
        "batch_size": 5_000,  # hands between regret dumps (larger → less variance, slower convergence)
        "n_cycles": 100,
        "regret_floor": 0.0,  # CFR+: clip negative regrets → 0 (keep 0.0)
        "p_cap": 2e-4,  # Pluribus reach-probability cap (1e-4–5e-4 works)

        # --- exploration ----------------------------------------------------
        "exploration": 0.05,  # ε-greedy while acting (0.05 → ≈ exploitative but still exploring)
        "exploration_decay": 0.999,  # per-hand multiplier (optional, see below)

        # --- averaging ------------------------------------------------------
        "avg_start": 40_000,  # start adding to average strategy after this many *hands*
        "avg_decay": 0.999,  # Robbins-Monro style weight on new iterates (optional)

        # --- numerics -------------------------------------------------------
        "regret_clip": 1e-12,  # ignore regrets whose |raw| < clip (speed, no ELO change)
        "strategy_clip": 1e-6,  # drop actions with prob < clip when saving (keeps files small)

        # --- diagnostics ----------------------------------------------------
        "checkpoint_every": 100_000,  # hands between serialisations
        "eval_every": 50_000,  # hands between exploitability check (if you have a solver)

    }

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
