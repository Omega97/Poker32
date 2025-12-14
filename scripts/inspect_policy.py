from pathlib import Path
from src.utils import inspect_policy


if __name__ == '__main__':

    # ------------------ CONFIGURATION ------------------
    # Model info
    _MODEL_NAME = "agentrl"
    _POLICY_PATH = Path(f"..\\models\\{_MODEL_NAME}.json")
    # _POLICY_PATH = Path(f"..\\models\\tournament_2025\\{_MODEL_NAME}.json")

    # --------------- INSPECT THE RESULTS ---------------
    # inspect_policy(file_path=_POLICY_PATH, show_proba=True)
    inspect_policy(file_path=_POLICY_PATH, show_proba=False)
