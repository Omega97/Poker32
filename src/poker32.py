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
POSITIONS = "SB", "BB"

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
            GAME_TREE[state_1] = s


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


def rewards(leaf: str, hole_cards: dict) -> Dict[str, int] | None:
    """
    Compute the monetary outcome for the two players.

    Returns
    -------
    (r0, r1) : dict
        r0 is the amount that the SB wins/loses; r1 is the same for the BB.
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

    # No player folded -> Showdown!
    hole_cards_values = [hole_cards["SB"], hole_cards["BB"]]
    if winner is None:
        hole_card_indexes = [RANKS.index(card) for card in hole_cards_values]
        best_card_id = min(hole_card_indexes)
        if best_card_id != max(hole_card_indexes):
            winner = hole_card_indexes.index(best_card_id)

    if winner == 0:
        return {"SB": bets[1], "BB": -bets[1]}
    elif winner == 1:
        return {"SB": -bets[0], "BB": bets[0]}
    else:
        return {"SB": 0, "BB": 0}


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
        self.branch = ""
        self.players = None
        self.reset()

    def get_button(self, offset=0) -> int | None:
        if self.button is None:
            return None
        return (self.button + offset) % 2

    def _move_button(self, button: int | None = None):
        if button is not None:
            self.button = button
        else:
            if self.get_button() is None:
                self.button = 0
            else:
                self.button = self.get_button(offset=1)

    def _init_deck(self, n_ranks=4):
        self.deck = list(RANKS) * n_ranks
        self.rng.shuffle(self.deck)

    def _get_hole_cards(self) -> List[str]:
        return self.hole_cards

    def _get_hole_card(self, player_id: int) -> str:
        return self.hole_cards[player_id]

    def _init_hole_cards(self, holes: Optional[List[str]] = None):
        if holes is not None:
            self.hole_cards = holes
        else:
            self.hole_cards = self.deck[:2]

    def _init_branch(self):
        self.branch = ""

    def _get_game_branch(self) -> str:
        return self.branch

    def _set_players(self, players):
        self.players = players

    def _get_players(self):
        return self.players

    def _get_player_to_act(self) -> int:
        n_turns = len(self._get_game_branch())
        return self.get_button(offset=n_turns)

    # --- State dictionaries -----------------------------------------------
    def _get_subjective_state(self, player_id: int) -> Dict:
        """
        Player 'player_id' is asking about the game state
        from their POV.
        """
        return {"player_id": player_id,
                "branch": self._get_game_branch(),
                "hole": self._get_hole_card(player_id),
                "legal_moves": self.get_legal_moves(),
                "rewards": self.get_rewards()}

    def _get_leaf_info(self, player_id: int) -> Dict:
        """Info for that player at the end of the game."""
        return {"player_id": player_id,
                "reward": self.get_player_reward(player_id),
                "position": self.get_player_position(player_id),
                "positions": self.get_positions(),
                "names": self.get_player_names(),
                "hole_cards": self._get_hole_cards(),
                "branch": self._get_game_branch(),
                "rewards": self.get_rewards(),
                }

    def _get_report(self) -> Dict:
        """Report with the results of the hand, and info about all players."""
        return {"names": self.get_player_names(),
                "position_ids": self.get_position_ids(),
                "positions": self.get_positions(),
                "hole_cards": self._get_hole_cards(),
                "branch": self._get_game_branch(),
                "rewards": self.get_rewards(),
                }

    # --- public API --------------------------------------------------------
    def get_player_names(self) -> list:
        return [p.get_name() for p in self.players]

    def get_position_ids(self):
        """Return the ids of SB and BB."""
        return {"SB": self.get_button(),
                "BB": self.get_button(offset=1)}

    def get_player_position(self, player_id: int) -> str:
        """Return the position (SB, BB) of a given player."""
        return POSITIONS[self.get_button(offset=player_id)]

    def get_positions(self):
        """Inverse of 'get_position_ids'."""
        return {i: self.get_player_position(i) for i in range(2)}

    def set_button(self, button: int):
        self.button = button % 2

    def reset(self, button: int | None = None, holes: Optional[Tuple[int, int]] = None):
        """
        button=0 -> Player 0 is SB (acts first), Player 1 is BB.
        button=1 -> Player 1 is SB, Player 0 is BB.
        """
        self._move_button(button)
        self._init_deck()
        self._init_hole_cards(holes)
        self._init_branch()

    def get_rgn_state(self):
        return self.rng

    def get_legal_moves(self) -> Tuple:
        """
        Get a tuple of all the legal moves in that state.
        Returns empty tuple if game is over.
        """
        key = self._get_game_branch()
        if key in GAME_TREE:
            branches = GAME_TREE[key]
            return tuple(sorted(s[-1] for s in branches))
        else:
            # For when checking for legal moves after game is over
            return tuple()

    def make_move(self, move: str):
        assert move in self.get_legal_moves()
        self.branch += move

    def is_game_over(self):
        return is_game_over(self._get_game_branch())

    def _get_hole_card_from_role(self, role: str) -> str:
        """Get hole card form role (SB, BB)"""
        return self._get_hole_cards()[self.get_position_ids()[role]]

    def _get_hole_cards_dict(self):
        """Return list of hole cards ordered relative to the button."""
        return {role:self._get_hole_card_from_role(role) for role in POSITIONS}

    def get_rewards(self) -> Dict[str, int]:
        """Return rewards of absolute positions."""
        return rewards(self._get_game_branch(), self._get_hole_cards_dict())

    def get_player_reward(self, player_id: int) -> int:
        position = self.get_player_position(player_id)
        return self.get_rewards()[position]

    def _pre_game_procedures(self):
        """Let the agents know that the game has started"""
        for i, player in enumerate(self._get_players()):
            root_info = {'position': self.get_player_position(i)}
            player.observe_root(root_info)

    def _main_game_loop(self):
        """Game loop."""
        while not self.is_game_over():
            player_to_act = self._get_player_to_act()
            legal_moves = self.get_legal_moves()

            # Let the agent choose an action, based on their perception of the game state
            policy = self._get_players()[player_to_act]
            action = policy(self._get_subjective_state(player_to_act))

            # Basic safety – fall back to a legal move if the agent is stupid
            if action not in legal_moves:
                action = next(iter(legal_moves))          # pick any legal move

            # Update game state
            self.make_move(action)

            # Notify all agents about the move
            for i, player in enumerate(self._get_players()):

                move_info = {"player_id": player_to_act,  # the player that made the move
                             "player_name": policy.name}

                player.broadcast_move(
                    move=action,
                    new_state=self._get_subjective_state(i),     # each agent gets its own view
                    move_info=move_info,
                )

    def _post_game_procedures(self):
        """Let the agents know about the result"""
        for player_id, player in enumerate(self._get_players()):
            player.observe_terminal(self._get_leaf_info(player_id))

    def play(self, players: List | Tuple):
        """
        Play a hand of Poker32 (Fresh deck, button moved by one).
        Output game report.
        """
        self.reset()
        self._set_players(players)
        self._pre_game_procedures()
        self._main_game_loop()
        self._post_game_procedures()
        return self._get_report()
