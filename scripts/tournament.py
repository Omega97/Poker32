# scripts/tournament.py
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.poker32 import Poker32
from src.rl_agent import load_rl_agent


def play_match(agent_a, agent_b, n_hands: int, seed: int):
    rng = random.Random(seed)
    game = Poker32(rng=rng)

    total_a = total_b = 0.0
    for _ in range(n_hands):
        result = game.play((agent_a, agent_b), verbose=False)
        rew_a, rew_b = result["rewards"]
        total_a += rew_a
        total_b += rew_b

    return total_a / n_hands, total_b / n_hands


def run_tournament(
    models_dir: str | Path = "models",
    n_hands: int = 100_000,
    seed: int = 42,
    figsize=(10, 8),
    cmap="RdYlGn",
    save_plot: str | None = "tournament_results.png"
):
    models_dir = Path(models_dir)
    model_files = sorted(models_dir.glob("*.pkl"))
    if not model_files:
        print("No .pkl models found in models/ folder!")
        return

    print(f"Loading {len(model_files)} agents...")
    agents = [load_rl_agent(p, verbose=True, training=False) for p in model_files]
    names = [p.stem for p in model_files]

    n = len(agents)
    results_matrix = np.zeros((n, n))

    print(f"\nRunning round-robin: {n} agents × {n_hands:,} hands per match\n")

    total_matches = n * (n - 1) // 2
    match_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            match_count += 1
            name_a, name_b = names[i], names[j]
            agent_a, agent_b = agents[i], agents[j]

            print(f"[{match_count:2d}/{total_matches}] {name_a:25} vs {name_b:25} ... ", end="", flush=True)

            ev_a, ev_b = play_match(agent_a, agent_b, n_hands, seed + i * n + j)

            results_matrix[i, j] = ev_a
            results_matrix[j, i] = ev_b

            print(f"{ev_a:+7.3f}  │  {ev_b:+7.3f}")

    # Fill diagonal
    np.fill_diagonal(results_matrix, 0.0)

    # === PLOT HEATMAP ===
    plt.figure(figsize=figsize)
    im = plt.imshow(results_matrix, cmap=cmap, vmin=-2.0, vmax=2.0, interpolation='nearest')

    plt.colorbar(im, label="Chips per hand (row vs column)", shrink=0.8)
    plt.title(f"Poker32 Round-Robin Tournament\n{n_hands:,} hands per match | "
              f"Total hands: {n_hands * total_matches * 2:,}")
    plt.xticks(np.arange(n), names, rotation=45, ha="right")
    plt.yticks(np.arange(n), names)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = results_matrix[i, j]
            color = "white" if abs(val) > 1.0 else "black"
            plt.text(j, i, f"{val:+.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    plt.xlabel("Opponent (column)")
    plt.ylabel("Player (row)")
    plt.tight_layout()

    if save_plot:
        plt.savefig(save_plot, dpi=200, bbox_inches='tight')
        print(f"\nHeatmap saved to {save_plot}")

    plt.show()

    # Also print clean text table
    print("\n" + "=" * 100)
    print("NUMERIC RESULTS (row beats column)")
    print("=" * 100)
    header = f"{'':25}" + "".join(f"{name:>12}" for name in names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row = f"{name:25}"
        for j in range(n):
            val = results_matrix[i, j]
            cell = "    —    " if i == j else f"{val:+8.3f}"
            row += f"{cell:>12}"
        print(row)
    print("=" * 100)


if __name__ == "__main__":
    run_tournament(
        models_dir="..\\models",
        n_hands=10_000,
        seed=42,
        save_plot="..\\data\\tournament_heatmap.png"
    )
