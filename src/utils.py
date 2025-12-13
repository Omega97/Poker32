import json
import math
from pathlib import Path
from typing import Dict
from src.poker32 import GAME_MOVES, RANKS


def _deserialize_infoset_key(s: str) -> tuple[str, str]:
    """Convert 'hole|branch' → (hole, branch)"""
    parts = s.split('|', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid serialized infoset key: {s}")
    return (parts[0], parts[1])


def inspect_policy(file_path: str | Path, show_proba: bool = True):
    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Trained for {data['games_played']:,} games")
    print(f"Config: {data['config']}")

    # Deserialize logits keys from string back to (hole, branch) tuples
    logits_serialized = data['logits']
    logits = {_deserialize_infoset_key(k): v for k, v in logits_serialized.items()}

    print(f"Total infosets learned: {len(logits)}")  # e.g., 416
    print()

    if show_proba:
        print("Softmax probabilities (min-normalized logits):")
    else:
        print("Logits")

    # Sort for consistent output
    rank_index = {rank: i for i, rank in enumerate(RANKS)}
    for (hole, branch), action_dict in sorted(logits.items(),
                                              key=lambda item: (rank_index[item[0][0]], item[0][1])):
        if not action_dict:
            continue  # safety

        s = f'"{branch}"'
        print(f'{hole} {s:7}', end=' →  ')

        # Note: reversed() was likely a mistake; preserve original action order
        # But we keep behavior identical to original for consistency
        logits_list = list(reversed(action_dict.values()))
        actions_list = list(reversed(action_dict.keys()))

        if show_proba:
            # Min-normalization + softmax
            min_l = min(logits_list)
            exps = [math.exp(logit - min_l) for logit in logits_list]
            total = sum(exps)
            probs = [e / total for e in exps]

            for move in GAME_MOVES:
                if move in actions_list:
                    i = actions_list.index(move)
                    act = actions_list[i]
                    p = probs[i]
                    print(f"{act}:{p:6.1%}", end="  ")
        else:
            # Max-normalization: best action = 0.0
            # max_l = max(logits_list)
            for act, l in zip(actions_list, logits_list):
                # normalized = l - max_l
                print(f"{act}:{l:5.2f}", end="  ")

        print()  # newline

    print("\nInspection complete.")


def round_floats(obj, decimals=3):
    """Recursively round all floats in a nested dict/list structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    else:
        return obj


def softmax(legal_moves: tuple[str, ...], action_logits: list) -> Dict[str, float]:
    """ Softmax with log-sum-exp trick. """
    max_logit = max(action_logits)
    shifted = [x - max_logit for x in action_logits]
    exps = [math.exp(x) for x in shifted]
    total = sum(exps)
    policy = {a: exp / total for a, exp in zip(legal_moves, exps)}
    return policy


def relu(x):
    return max(0., x)


def maturity(logits, k=3.):
    """Value in [0, 1], indicates how saturated are the numbers in the list."""
    if len(logits):
        logits = [relu(math.tanh(x/k)) for x in logits]
        return sum(logits) / len(logits)
    else:
        return 0.


def safety_check(policy_path: Path, model_name: str):
    if policy_path.exists() and policy_path.stat().st_size:
        input(f"{policy_path.name} already exists – proceed anyway?")
    else:
        print(f'Training "{model_name}" ({policy_path})')
