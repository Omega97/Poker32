# scripts/one_vs_all.py
"""
Let one model learn by playing against every other frozen model in models/
Usage:
    python one_vs_all.py
"""
import random
from pathlib import Path
from src.poker32 import Poker32
from src.agents.rl_agent import load_rl_agent, AgentRL


class OneVsAllTrainer:
    def __init__(
        self,
        target_name: str,
        n_hands_each: int = 100_000,
        models_dir: str | Path = "models",
        seed: int = 42,
    ):
        self.target_name = target_name
        self.n_hands_each = n_hands_each
        self.models_dir = Path(models_dir)
        self.seed = seed

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

    def get_target_path(self) -> Path:
        return self.models_dir / f"{self.target_name}.json"

    def get_opponent_name(self, path: Path) -> str:
        return path.stem

    def load_target_agent(self) -> AgentRL:
        target_path = self.get_target_path()
        if not target_path.exists():
            raise FileNotFoundError(f"Target agent not found: {target_path}")

        target = load_rl_agent(target_path, training=True, verbose=True)
        target.games_played = 0  # reset session counter
        print(f"Loaded {self.target_name} – training ACTIVE\n")
        return target

    def load_opponents(self) -> list[AgentRL]:
        opponents = []
        for path in self.models_dir.glob("*.json"):
            if path.stem == self.target_name:
                continue
            opp = load_rl_agent(path, training=False, verbose=False)
            opp.name = self.get_opponent_name(path)
            opponents.append(opp)

        if not opponents:
            raise RuntimeError(f"No opponent models found in {self.models_dir}/ (excluding {self.target_name})")

        print(f"Found {len(opponents)} frozen opponents – training OFF\n")
        return opponents

    def train(self, n_hands_period=10_000):
        random.seed(self.seed)
        rng = random.Random(self.seed)

        target = self.load_target_agent()
        opponents = self.load_opponents()

        if not opponents:
            raise RuntimeError("No opponents to train against!")

        # Preserve total training budget: same as before
        total_hands = self.n_hands_each * len(opponents)

        game = Poker32(rng=rng)
        total_reward = 0.0

        print(f"► {self.target_name} vs RANDOM opponent (total {total_hands:,} hands)")

        for hand in range(1, total_hands + 1):
            # Pick random opponent
            opp = rng.choice(opponents)

            # Randomize seating
            result = game.play((target, opp), verbose=False)
            reward = result["rewards"][1]

            total_reward += reward

            # Apply updates immediately
            target._apply_accumulated_updates()

            # Periodic reporting
            if hand % n_hands_period == 0 or hand == total_hands:
                avg_ev = total_reward / hand
                print(f"  Hand {hand:,}  Avg EV: {avg_ev:+.3f}")

        print(f"\nTraining complete! Total hands: {total_hands:,}")
        print(f"Final average EV: {total_reward / total_hands:+.3f}\n")

        # Save updated target
        target.save(self.get_target_path())
        print(f"Updated {self.target_name} saved – total new games: {target.games_played:,}")


if __name__ == "__main__":
    # ------------------ CONFIGURATION ------------------
    TARGET_NAME = "new"
    N_HANDS_EACH = 100_000
    SEED = 42
    MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
    # --------------------------------------------------

    # Run one-vs-all training
    trainer = OneVsAllTrainer(
        target_name=TARGET_NAME,
        n_hands_each=N_HANDS_EACH,
        models_dir=MODEL_DIR,
        seed=SEED,
    )
    trainer.train()
