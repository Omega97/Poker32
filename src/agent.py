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
            actor = "You" if actor_id == 0 else "Opponent"
            print(f"  → {actor} played: {move}  |  branch → {new_state['branch'] or 'root'}")


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

    def observe_root(self, state: dict):
        if self.verbose:
            print("\n" + "="*60)
            print(f"     NEW HAND — You are {self.name}")
            print("="*60)

    def choose_action(self, state: dict) -> str:
        hole = state["hole"]
        branch = state["branch"] or "root"
        legal_moves = "".join(state["legal_moves"])
        player_id = state.get("player_id", 0)

        print()
        print(f"> Your hole card: {hole}")
        print(f'Current betting sequence: "{branch}"')
        # print(f"Pot size: ~{pot} chips")
        # print(f"Legal actions: {', '.join(legal_moves)}")

        # Show action meaning
        meaning = {
            'f': "fold",
            'c': "call/check",
            'R': "raise (to 4)",
            'D': "double raise (to 8)",
            'T': "triple raise (to 16)",
            'Q': "quadruple all-in (to 32)"
        }
        # print("Actions: " + "  ".join(f"{a}={meaning.get(a,a)}" for a in legal_moves))

        while True:
            choice = input(f"\nYour move [{'/'.join(legal_moves)}]: ").strip().lower()
            if choice in legal_moves.lower():
                print(f"You chose: {choice} ({meaning.get(choice, choice)})")
                return choice
            else:
                print(f"'{choice}' is an invalid move! Legal: {legal_moves}")

    def observe_terminal(self, state: dict):
        rewards = state["rewards"]
        my_reward = rewards[0]  # we are always player 0 when playing
        opp_card = state.get("opp_card")  # optional, if you expose it

        print("\n" + "="*60)
        print("HAND OVER")
        if my_reward > 0:
            print(f"You WON +{my_reward} chips!")
        elif my_reward < 0:
            print(f"You lost {my_reward} chips")
        else:
            print("Split pot")
        if opp_card:
            print(f"Opponent had: [{opp_card}]")
        print("="*60 + "\n")
