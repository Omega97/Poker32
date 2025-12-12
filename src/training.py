import pathlib
from typing import Optional
import random
from src.poker32 import Poker32


class Poker32Trainer:
    def __init__(self,
                 agent_class,
                 file_path: pathlib.Path,
                 config: dict,
                 name: str = 'training_agent',
                 rng: Optional[random.Random] = None):
        self.agent_class = agent_class
        self.file_path = file_path
        self.config = config
        self.name = name
        self.rng = rng
        self.agent = None
        self.game = None

    def run(self, **kwargs):
        """Main training entry point. Accepts optional config overrides via kwargs."""
        # Create a temporary config by shallow-copying self.config and updating with kwargs
        train_config = self.config.copy()
        train_config.update(kwargs)

        self._initialize_agent(train_config)
        self._setup_game()
        self._print_training_info(train_config)
        self._training_loop(train_config)
        self._save_agent()

    def _initialize_agent(self, config):
        """Load existing agent or create a new one using the given config."""
        if self.file_path.exists():
            print(f"Loading existing policy from {self.file_path}")
            self.agent = self.agent_class.load(
                str(self.file_path),
                config=config,
                name=self.name,
                training=True
            )
            print(f"Resuming from {self.agent.games_played:,} games")
        else:
            print("No existing policy found → starting from scratch")
            self.agent = self.agent_class(
                config=config,
                name=self.name,
                rng=self.rng,
                training=True
            )

    def _setup_game(self):
        """Instantiate the game environment."""
        self.game = Poker32(rng=self.rng)

    def _print_training_info(self, config):
        """Print initial training summary using the effective config."""
        total_games = config['batch_size'] * config['n_cycles']
        print("Starting Poker32 RL training (additive logit, T=1.0)")
        print(f"Target: {total_games:,} games\n")

    def _training_loop(self, config):
        """Run the main training cycles using the effective config."""
        for cycle in range(config["n_cycles"]):
            self._play_batch(config)

            maturity = self.agent.get_maturity()
            print(f"\rCycle {cycle+1}/{config['n_cycles']} | "
                  f"Games played: {self.agent.games_played:,} | "
                  f"mat={maturity:.2%}", end='', flush=True)
        print()

    def _play_batch(self, config):
        """Play one batch of games using the effective config."""
        for _ in range(config["batch_size"]):
            self.game.play((self.agent, self.agent))

    def _save_agent(self):
        """Save the trained agent to disk."""
        self.agent.save(self.file_path)
        print(f"Training complete! Policy saved.")
