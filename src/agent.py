# src/agents.py
import random
from pathlib import Path
from typing import Dict
import json
import matplotlib.pyplot as plt
from src.poker32 import init


# Initialize game tree once at import time
init()


# Infoset key: (hole_card, betting_branch)
# e.g. ("A", "cR") or ("K", "RRf")
InfosetKey = tuple[str, str]


ACTIONS = {
    'f': "🚫 fold",
    'c': "🤝 call/check",
    'R': "📈 raise",
    'D': "‼️ double raise",
    'T': "🚀 triple raise",
    'Q': "💥 all-in"
}


class Agent:
    """
    Base class for all Poker32 agents.
    """
    def __init__(self, rng: random.Random | None = None, verbose=True,
                 momentum=0.9, name='Agent'):
        self.rng = rng or random.Random()
        self.verbose = verbose
        self.update_momentum: Dict[InfosetKey, Dict[str, float]] = {}  # persistent velocity
        self.momentum = momentum
        self.name = name

    def get_name(self) -> str:
        return self.name

    def choose_action(self, state: dict) -> str:
        """Return a legal action given the current game state."""
        raise NotImplementedError

    def __call__(self, state: dict):
        return self.choose_action(state)

    def observe_root(self, state: dict):
        """
        Called at the beginning of each hand so the agent
        knows that the game has begun.
        """
        if self.verbose:
            print(f'> "{self.name}" observes {state}')

    def observe_terminal(self, state: dict):
        """
        Called at the end of each hand so the agent can learn.
        Default: do nothing (for random or fixed agents).
        """
        if self.verbose:
            print(f'> "{self.name}" observes {state}')

    def save(self, filepath: str | Path):
        """Optional persistence."""
        pass

    def broadcast_move(self, move: str, new_state: dict, move_info: dict):
        """
        Called on ALL agents (including the one who moved) after their action).
        Useful for: logging, UI updates, debugging, or future multi-agent learning.
        """
        if self.verbose:
            branch = new_state["branch"]
            player_name = move_info["player_name"]
            print(f'> "{self.name}": "{player_name}" played {ACTIONS[move]}  -> "{branch}"')


class RandomAgent(Agent):
    """Purely random legal moves."""
    def choose_action(self, state: dict) -> str:
        """Random action"""
        legal = state['legal_moves']
        action = self.rng.choice(list(legal))
        if self.verbose:
            print(f'> "{self.name}": {legal} -> {action}')
        return action


class HumanAgent(Agent):
    """
    Human player for Poker32.
    Shows current hand, branch, legal moves, and asks for input.
    """

    def __init__(self, name: str = "Human", verbose: bool = True, log_path: str | Path | None = None):
        super().__init__(rng=None, verbose=verbose)
        self.name = name
        self.cumulative_chips = 0
        self.hands_played = 0
        self.rewards_log: list[int] = []  # reward per hand

        # Optional persistent log file
        self.log_path = Path(log_path) if log_path else None
        if self.log_path and self.log_path.exists():
            self._load_rewards_log()

    def observe_root(self, state: dict):
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"> NEW HAND")
            print(f"> You are {self.name}")
            if state.get("position") == 0:
                print("You are the SB 🔹")
            else:
                print("> You are the BB 🟦")
            print()
            print(' ----- ACTION BEGINS -----')

    def choose_action(self, state: dict) -> str:
        hole = state["hole"]
        branch = state["branch"] or "root"
        legal_moves = state["legal_moves"]   # e.g. ('f','c','R','D', …)

        print(f"> Your hole card: {hole}🀄️")
        print(f'> Branch: "{branch}"')

        while True:
            choice = input(f"Your move [{'/'.join(legal_moves)}]: ").strip().lower()
            # compare case-insensitively, but send the **real** move back
            for mv in legal_moves:
                if choice == mv.lower():
                    return mv          # <-- upper-case
            print(f"'{choice}' is an invalid move! Legal: {''.join(legal_moves)}")

    def save_rewards_log(self, filepath: str | Path | None = None):
        """Save rewards history to JSON."""
        fp = Path(filepath or self.log_path or f"{self.name.lower()}_rewards.json")
        data = {
            "agent_name": self.name,
            "rewards": self.rewards_log,
            "total_hands": len(self.rewards_log),
            "cumulative": sum(self.rewards_log)
        }
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)

    def observe_terminal(self, state: dict):

        my_reward = state["reward"]

        # Track reward
        self.rewards_log.append(my_reward)
        self.cumulative_chips += my_reward
        self.hands_played += 1

        # Print outcome
        if self.verbose:
            print("\n ----- HAND OVER -----")

            print(state)

            if my_reward > 0:
                print(f"🏆  You WON +{my_reward} 🟡")
            elif my_reward < 0:
                print(f"💸  You lost {abs(my_reward)} 🟡")
            else:
                print("⚖️  Split pot")

            hole_cards = state.get("hole_cards")
            if hole_cards:
                print(f"🃏  Hole cards {hole_cards}")

            avg = self.cumulative_chips / self.hands_played
            print(f"📊  Cumulative: {self.cumulative_chips:+} 🟡  "
                  f"Average: {avg:+.2f} 🟡 / hand")
            print("=" * 60 + "\n")

        # Auto-save if log_path is set
        if self.log_path:
            self.save_rewards_log()

    # ----- Data -----

    def _load_rewards_log(self):
        """Load past rewards from JSON file."""
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
                self.rewards_log = data.get("rewards", [])
                self.cumulative_chips = sum(self.rewards_log)
                self.hands_played = len(self.rewards_log)
                print(f"Loaded {self.hands_played} past hands from {self.log_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load log file: {e}")
            self.rewards_log = []

    def plot_cumulative_returns(self, figsize=(10, 5)):
        """
        Plot cumulative rewards over time using matplotlib.
        """
        if not self.rewards_log:
            print("No rewards to plot!")
            return

        cumsum = [0]
        total = 0
        for r in self.rewards_log:
            total += r
            cumsum.append(total)
        x_ = list(range(len(cumsum)))

        plt.figure(figsize=figsize)
        plt.plot(x_, cumsum, color="tab:blue", linewidth=1.2)
        plt.title(f"{self.name} — Cumulative Chips Over Time")
        plt.xlabel("Hand #")
        plt.ylabel("Cumulative Chips")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.show()
