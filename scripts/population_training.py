# scripts/population_training.py
import random
from pathlib import Path
from src.agents.rl_agent import AgentRL
from src.poker32 import Poker32


class PopulationTrainer:
    DEFAULT_CONFIG = {
        "learning_rate": 0.05,
        "temperature": 1.0,
        "momentum": 0.9,
        "logit_range": 10.0,
        "init_range": 0.2,
        "n_epochs": 1,
        "n_cycles": 1,
    }

    def __init__(
        self,
        n_agents: int = 8,
        n_hands: int = 1_000_000,
        config: dict | None = None,
        save_dir: str | Path = "models/population",
        rng: random.Random | None = None,
    ):
        self.n_agents = n_agents
        self.n_hands = n_hands
        self.save_dir = Path(save_dir)
        self.rng = rng

        # Merge config
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        # Prepare directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated in _initialize_agents
        self.agents: list[AgentRL] = []

    def get_agent_name(self, i: int) -> str:
        """Return the name of agent at index i (1-based naming)."""
        return f"tournament_{i + 1}"

    def _initialize_agents(self):
        """Load existing agents or create new ones if missing."""
        self.agents.clear()
        for i in range(self.n_agents):
            name = self.get_agent_name(i)
            path = self.save_dir / f"{name}.json"

            if path.exists():
                agent = AgentRL.load(
                    path,
                    training=True,
                    config=self.config,
                    rng=random.Random(self.rng.random()),
                )
                agent.name = name
                print(f"  → Loaded {name} ({agent.games_played:,} games)")
            else:
                agent = AgentRL(
                    rng=random.Random(self.rng.random()),
                    config=self.config,
                    training=True,
                )
                agent.name = name
                print(f"  → Created {name}")

            self.agents.append(agent)

    def train(self):
        """Run population self-play training."""
        self._initialize_agents()

        game = Poker32(rng=self.rng)
        print(f"\nStarting population self-play: {self.n_hands:,} hands")

        for hand in range(1, self.n_hands + 1):
            a1, a2 = random.sample(self.agents, 2)
            game.play((a1, a2))

            if hand % 50_000 == 0 or hand == self.n_hands:
                print(f"Hand {hand:,}")

        # Save all agents
        print("\nTraining complete! Saving final population...")
        for i, agent in enumerate(self.agents):
            path = self.save_dir / f"{self.get_agent_name(i)}.json"
            agent.save(path)
            print(f"  → {agent.name} saved → {agent.games_played:,} games")

        print(f"\nPopulation training finished!")
        print(f"All agents in: {self.save_dir.resolve()}")


if __name__ == "__main__":
    # ------------------ CONFIGURATION ------------------
    _N_HANDS = 5_000_000
    _N_AGENTS = 3

    _RNG = random.Random(0)
    _DIR_PATH = "../models/tournament_2025"
    _CONFIG = {"learning_rate": 0.05,  # step length for each spot
               "temperature": 1.0,  # modifier for the policy sampling
               "init_range": 0.1,  # initial range for the logits
               "logit_range": 20,  # logits are capped between +/- this value
               "momentum": 0.995,  # decay on accumulated rewards and counts
               "damping": 0.995,  # attract the logits towards zero
               }

    # ---------------------------------------------------

    # Run population-based tournament
    trainer = PopulationTrainer(
        n_agents=_N_AGENTS,
        n_hands=_N_HANDS,
        config=_CONFIG,
        save_dir=_DIR_PATH,
        rng=_RNG,
    )
    trainer.train()
