# src/poker32.py
from __future__ import annotations
import random
from typing import Tuple, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANKS = "AKQJT98765432"
N_RANKS = len(RANKS)
N_SUITS = 4

# Fast rank → index lookup (A=0 … 2=12)
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
IDX_TO_RANK = {i: r for r, i in RANK_TO_IDX.items()}

# Stake ladder: 1, 2, 4, 8, 16, 32
N_RAISES = 4
STAKE_LADDER = tuple(2 ** i for i in range(N_RAISES + 2))

# Action encoding
GAME_MOVES = "fcRDTQ"  # fold, call, Raise, Double, Triple, Quadruple

# Global containers populated by init()
TERMINAL_NODES: Tuple[str, ...] | None = None
GAME_TREE: Dict[str, frozenset[str]] | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ordered_compositions(n):
    """
    Return every ordered list of positive integers that sum to n.
    Used to enumerate all legal betting sequences.
    """
    if n == 0:
        return [[]]  # base case: empty composition
    out = []
    for first in range(1, n + 1):  # choose first term
        for rest in ordered_compositions(n - first):
            out.append([first] + rest)  # prepend first to every suffix
    return out


# --------------------------------------------------------------------------- #
# Game-tree construction
# --------------------------------------------------------------------------- #
def _generate_terminal_nodes():
    global TERMINAL_NODES, N_RAISES

    if TERMINAL_NODES is not None:
        return

    TERMINAL_NODES = ['f', 'cc']

    for n in range(1, N_RAISES+1):
        for c in ordered_compositions(n):
            branch = "".join([GAME_MOVES[i+1] for i in c])
            TERMINAL_NODES.append(branch + 'f')
            TERMINAL_NODES.append(branch + 'c')
            TERMINAL_NODES.append('c' + branch + 'c')
            TERMINAL_NODES.append('c' + branch + 'f')

    TERMINAL_NODES = tuple(TERMINAL_NODES)


def _init_game_tree():
    global TERMINAL_NODES, GAME_TREE
    assert TERMINAL_NODES is not None
    if GAME_TREE is not None:
        return

    GAME_TREE = dict()

    for branch in TERMINAL_NODES:
        for i in range(len(branch)):
            state_1 = branch[:i]
            state_2 = branch[:i+1]
            s = GAME_TREE.get(state_1, set())
            s.add(state_2)
            GAME_TREE [state_1] = s


def init():
    _generate_terminal_nodes()
    _init_game_tree()


# --------------------------------------------------------------------------- #
# Reward engine
# --------------------------------------------------------------------------- #
def is_game_over(leaf):
    """Return True if `leaf` is a terminal betting sequence."""
    if leaf == 'cc':
        return True
    elif leaf == '':
        return False
    elif leaf == GAME_MOVES[1]:
        return False
    else:
        return leaf[-1] in set(GAME_MOVES[:2])


def rewards(leaf: str, holes: List[str]) -> Tuple[int, int] | None:
    """
    Compute the monetary outcome for the two players.

    Returns
    -------
    (p0, p1) : tuple[int, int]
        p0 is the amount player 0 wins/loses; p1 is the same for player 1.
    """
    if not is_game_over(leaf):
        return None

    bets = [1, 2]  # blinds
    winner = None
    for i, move in enumerate(leaf):
        player_id = i % 2
        move_id = GAME_MOVES.index(move)
        if move_id == 0:  # fold
            winner = 1 - player_id
        elif move_id == 1:  # call
            bets[player_id] = max(bets)
        else:  # raise
            bets[player_id] = max(bets) * 2 ** (move_id - 1)

    # No player folded
    if winner is None:
        hole_ids = [RANKS.index(card) for card in holes]
        best_card_id = min(hole_ids)
        if best_card_id != max(hole_ids):
            winner = hole_ids.index(best_card_id)

    if winner == 0:
        return bets[1], -bets[1]
    elif winner == 1:
        return -bets[0], bets[0]
    else:
        return 0, 0


# --------------------------------------------------------------------------- #
# Game class
# --------------------------------------------------------------------------- #
class Poker32:
    """
    Heads-up one-card betting game with stake ladder {1,2,4,8,16,32}.
    Settlement equals the final stake (not a pot count).
    """
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.button: int | None = None
        self.deck: List[str] = []
        self.hole_cards: List[str] = []
        self.branch = ''
        self.reset()

    def _init_button(self, button: int|None = None):
        if button is not None:
            self.button = button
        else:
            self.button = self.rng.randint(0, 1)

    def _init_deck(self):
        self.deck = list(RANKS) * 4
        self.rng.shuffle(self.deck)

    def _init_holes(self, holes: Optional[List[str]] = None):
        if holes is not None:
            self.hole_cards = holes
        else:
            self.hole_cards = self.deck[:2]

    def _init_branch(self):
        self.branch = ''

    def _get_holes(self) -> List[str]:
        return self.hole_cards

    def _get_game_branch(self) -> str:
        return self.branch

    # --- public API --------------------------------------------------------
    def reset(self, button: int|None = None, holes: Optional[Tuple[int, int]] = None):
        """
        button=0 -> Player 0 is SB (acts first), Player 1 is BB.
        button=1 -> Player 1 is SB, Player 0 is BB.
        """
        self._init_button(button)
        self._init_deck()
        self._init_holes(holes)
        self._init_branch()

    def get_rgn_state(self):
        return self.rng

    def get_subjective_state(self, player_id: int) -> Dict:
        return {"branch": self._get_game_branch(),
                "hole": self.hole_cards[player_id],
                "legal_moves": self.get_legal_moves(),
                "player_id": player_id,
                "rewards": self.get_rewards()}

    def get_legal_moves(self) -> Tuple:
        branches = GAME_TREE[self._get_game_branch()]
        return tuple(sorted(s[-1] for s in branches))

    def make_move(self, move: str):
        assert move in self.get_legal_moves()
        self.branch += move

    def is_game_over(self):
        return is_game_over(self._get_game_branch())

    def get_rewards(self) -> Tuple[int, int]:
        return rewards(self._get_game_branch(), self._get_holes())

    # Add this improved version to poker32.py (or keep separate)
    def play(
        self,
        players: Tuple,
        *,
        verbose: bool = False
    ):
        """
        Play a hand of Poker32.
        fresh deck + random button
        """
        self.reset()

        # Let the agents know that the game has started
        for i, player in enumerate(players):
            root_info = {'position': i}
            player.observe_root(root_info)

        # Game loop
        while not self.is_game_over():
            player_to_act = len(self._get_game_branch()) % 2
            legal_moves = self.get_legal_moves()

            # Let the agent choose an action, based on their perception of the game state
            policy = players[player_to_act]
            action = policy(self.get_subjective_state(player_to_act))

            # Basic safety – fall back to a legal move if the agent is stupid
            if action not in legal_moves:
                action = next(iter(legal_moves))          # pick any legal move
            self.make_move(action)

            if verbose:
                print(action)
                hole_str = f"p0:{self.hole_cards[0]} p1:{self.hole_cards[1]}"
                print(f"{'SB' if player_to_act == (1-self.button) else 'BB'} "
                      f"({player_to_act}) plays {action} → {self._get_game_branch()}   {hole_str}")

        rewards = self.get_rewards()

        if verbose:
            winner = 0 if rewards[0] > 0 else (1 if rewards[1] > 0 else "split")
            print(f"GAME OVER → {self._get_game_branch()} | rewards {rewards} | winner: {winner}")
            print(f"Holes: p0={self.hole_cards[0]}  p1={self.hole_cards[1]}")

        # Let the agents know about the result
        for i, player in enumerate(players):
            leaf_info = {"position": i,
                         "branch": self._get_game_branch(),
                         "rewards": self.get_rewards(),
                         "reward": self.get_rewards()[i],
                         "opp_card": self.hole_cards[1 - i],
                         }
            player.observe_terminal(leaf_info)

        report = {"rewards": rewards,
                  "branch_info": self._get_game_branch(),
                  "hole_cards": self.hole_cards}

        return report
