"""
scripts/pipelines/pipeline_agentrl.py
Train AgentUCB with a long schedule and progressive damping.
"""
import random
from pathlib import Path
from src.utils import inspect_policy, safety_check
from src.training import Poker32Trainer
from src.agents.rl_agent import AgentRL


if __name__ == '__main__':
    # ------------------ CONFIGURATION ------------------
    _MODEL_NAME = "agentrl"
    _AGENT_CLASS = AgentRL
    _POLICY_PATH = Path(f"..\\..\\models\\{_MODEL_NAME}.json")
    _RNG = random.Random(42)
    _CONFIG = {"learning_rate": 0.05,  # step length for each spot
               "init_range": 0.1,  # initial range for the logits
               "logit_range": 20,  # logits are capped between +/- this value
               "momentum": 0.9,  # decay on accumulated rewards and counts
               "damping": 0.9,  # attract the logits towards zero
               "batch_size": 5_000,  # number of hands per cycle
               "n_cycles": 50}  # number of updates

    # ---------------- TRAINING PIPELINE ----------------
    safety_check(_POLICY_PATH, _MODEL_NAME)
    trainer = Poker32Trainer(agent_class=_AGENT_CLASS, file_path=_POLICY_PATH, config=_CONFIG, rng=_RNG)
    trainer.run(damping=0.9)
    trainer.run(damping=0.99)
    trainer.run(damping=0.999)

    # --------------- INSPECT THE RESULTS ---------------
    inspect_policy(file_path=_POLICY_PATH, show_proba=True)  # <- set to 'False' to show logits instead
