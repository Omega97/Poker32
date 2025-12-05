# scripts/tournament.py
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.poker32 import Poker32
from src.rl_agent import load_rl_agent


def play_match(agent_a, agent_b, n_hands: int, rng:random.Random | None = None):
    """
    Play n_hands between the agents.
    """
    if rng is None:
        rng = random.Random()

    game = Poker32(rng=rng)

    total_a = total_b = 0.0
    for _ in range(n_hands):
        result = game.play((agent_a, agent_b))

        rewards = result["rewards"]
        positions = result["positions"]

        sorted_rewards = [rewards[positions[i]] for i in range(2)]
        total_a += sorted_rewards[0]
        total_b += sorted_rewards[1]

    return total_a / n_hands, total_b / n_hands


def run_tournament(
    models_dir: str | Path = "models",
    n_hands: int = 100_000,
    rng: random.Random | None = None,
    figsize=(10, 8),
    cmap="RdYlGn",
    name_length=20,
    save_plot: str | Path | None = "tournament_results.png"
):
    """
    Note: Poker instance is deterministic.
    """
    models_dir = Path(models_dir)
    # 🔁 CHANGE: look for .json instead of .pkl
    model_files = sorted(models_dir.glob("*.json"))
    if not model_files:
        print(f"No .json models found in {models_dir}/ folder!")
        return

    print(f"Loading {len(model_files)} agents...")
    agents = [load_rl_agent(p, rng=rng, verbose=True, training=False) for p in model_files]
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

            print(f"[{match_count:2d}/{total_matches}] {name_a:{name_length}} vs {name_b:{name_length}} ... ", end="", flush=True)

            rng = random.Random(i * n + j)
            ev_a, ev_b = play_match(agent_a, agent_b, n_hands, rng=rng)

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
    header = f"{'':{name_length}}" + "".join(f"{name:>12}" for name in names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row = f"{name:{name_length}}"
        for j in range(n):
            val = results_matrix[i, j]
            cell = "    —    " if i == j else f"{val:+8.3f}"
            row += f"{cell:>12}"
        print(row)
    print("=" * 100)


if __name__ == "__main__":
    # ------------------ CONFIGURATION ------------------
    _MODELS_DIR = Path("..") / "models"
    # _MODELS_DIR = Path("..") / "models" / "tournament_2025"

    _PLOT_PATH = Path("..") / "data" / "tournament_heatmap.png"
    _N_HAND = 5_000
    _RNG = random.Random(0)
    # ---------------------------------------------------

    # Run tournament
    run_tournament(
        models_dir=_MODELS_DIR,
        n_hands=_N_HAND,
        rng=_RNG,
        save_plot=_PLOT_PATH,
    )
