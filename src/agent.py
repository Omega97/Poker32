# src/agents.py
import random
from pathlib import Path
from typing import Dict
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
    def __init__(self, rng: random.Random | None = None, verbose=True, momentum=0.9):
        self.rng = rng or random.Random()
        self.verbose = verbose
        self.update_momentum: Dict[InfosetKey, Dict[str, float]] = {}  # persistent velocity
        self.momentum = momentum

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
            print(state)

    def observe_terminal(self, state: dict):
        """
        Called at the end of each hand so the agent can learn.
        Default: do nothing (for random or fixed agents).
        """
        if self.verbose:
            print(state)

    def save(self, filepath: str | Path):
        """Optional persistence."""
        pass

    def broadcast_move(self, move: str, new_state: dict, actor_id: int):
        """
        Called on ALL agents (including the one who moved) after their action).
        Useful for: logging, UI updates, debugging, or future multi-agent learning.
        """
        if self.verbose:
            print(f'> {actor_id} played "{ACTIONS[move]}"  |  branch → {new_state["branch"] or "root"}')


class RandomAgent(Agent):
    """Purely random legal moves."""
    def choose_action(self, state: dict) -> str:
        """Random action"""
        legal = state['legal_moves']
        action = self.rng.choice(list(legal))
        if self.verbose:
            print(f'{legal} -> {action}')
        return action


class HumanAgent(Agent):
    """
    Human player for Poker32.
    Shows current hand, branch, legal moves, and asks for input.
    """

    def __init__(self, name: str = "Human", verbose: bool = True):
        super().__init__(rng=None, verbose=verbose)
        self.name = name
        # ---------- NEW: track performance ----------
        self.cumulative_chips = 0
        self.hands_played = 0
        # -------------------------------------------

    def observe_root(self, state: dict):
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"> NEW HAND")
            print(f"> You are {self.name}")
            if state.get("position") == 0:
                print("🔹 You are the Small Blind (act first)")
            else:
                print("🟦 You are the Big Blind (act second)")
            print()
            print(' ----- ACTION BEGINS -----')

    def choose_action(self, state: dict) -> str:
        hole = state["hole"]
        branch = state["branch"] or "root"
        legal_moves = state["legal_moves"]   # e.g. ('f','c','R','D', …)

        print(f"🀄️  Your hole card: {hole}")
        print(f'> Current betting sequence: "{branch}"')

        while True:
            choice = input(f"Your move [{'/'.join(legal_moves)}]: ").strip().lower()
            # compare case-insensitively, but send the **real** move back
            for mv in legal_moves:
                if choice == mv.lower():
                    return mv          # <-- upper-case
            print(f"'{choice}' is an invalid move! Legal: {''.join(legal_moves)}")

    # ------------------------------------------------------------------
    # NEW: updated terminal observer with cumulative stats
    # ------------------------------------------------------------------
    def observe_terminal(self, state: dict):
        rewards = state["rewards"]
        player_id = state["position"]
        my_reward = rewards[player_id]
        opp_card = state.get("opp_card")

        # ---- update trackers ----
        self.cumulative_chips += my_reward
        self.hands_played += 1
        # -------------------------

        print("\n ----- HAND OVER -----")
        if my_reward > 0:
            print(f"🏆  You WON +{my_reward} 🟡")
        elif my_reward < 0:
            print(f"💸  You lost {abs(my_reward)} 🟡")
        else:
            print("⚖️  Split pot")

        if opp_card:
            print(f"🃏  Opponent had {opp_card}")

        # ---- show running stats ----
        avg = self.cumulative_chips / self.hands_played
        print(f"📊  Cumulative: {self.cumulative_chips:+} 🟡  "
              f"Average: {avg:+.2f} 🟡 / hand")
        print("=" * 60 + "\n")
