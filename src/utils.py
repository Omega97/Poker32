import pickle
import math
from pathlib import Path
from src.poker32 import GAME_MOVES, RANKS


def inspect_policy(file_path: str | Path, show_proba: bool = True):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"Trained for {data['games_played']:,} games")
    print(f"Config: {data['config']}")
    print(f"Total infosets learned: {len(data['logits'])}")  # 416
    print()

    if show_proba:
        print("Softmax probabilities (min-normalized logits):")
    else:
        print("Max-normalized logits (best action = 0.00):")

    # Sort for consistent output
    rank_index = {rank: i for i, rank in enumerate(RANKS)}
    for (hole, branch), action_dict in sorted(data['logits'].items(),
                                              key=lambda item: (rank_index[item[0][0]], item[0][1])
                                              ):
        if not action_dict:
            continue  # safety

        s = f'"{branch}"'
        print(f'{hole} {s:7}', end=' →  ')

        logits = list(reversed(action_dict.values()))
        actions = list(reversed(action_dict.keys()))

        if show_proba:
            # Min-normalization + softmax
            min_l = min(logits)
            exps = [math.exp(logit - min_l) for logit in logits]
            total = sum(exps)
            probs = [e / total for e in exps]

            for move in GAME_MOVES:
                if move in actions:
                    i = actions.index(move)
                    act = actions[i]
                    p = probs[i]
                    print(f"{act}:{p:6.1%}", end="  ")
        else:
            # Max-normalization: best action = 0.0
            max_l = max(logits)
            for act, l in zip(actions, logits):
                normalized = l - max_l
                print(f"{act}:{normalized:5.1f}", end="  ")

        print()  # newline
    print("\nInspection complete.")
