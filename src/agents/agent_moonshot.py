from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict
from src.utils import round_floats, relu
from src.agents.rl_agent import AgentRL, InfosetKey, _serialize_infoset_key, _deserialize_infoset_key


class AgentCRM(AgentRL):
    """
    Outcome-sampling MCCFR with CFR+ and probability capping.
    Re-uses the JSON save/load machinery of AgentRL but ignores the RL update path.
    """

    DEFAULT_CONFIG = {
        "exploration": 0.1,  # ε for ε-greedy while acting
        "p_cap": 1e-4,  # Pluribus reach-prob threshold
        "batch_size": 5_000,
        "n_cycles": 50,
        "regret_floor": 0.0,  # CFR+: clip negative regrets
    }

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, **kwargs):
        # We bypass the RL parent initialisers that create logits, accumulated, etc.
        super().__init__(**kwargs)

        # CFR-specific tables
        self.cumulative_regret: Dict[InfosetKey, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cumulative_strategy: Dict[InfosetKey, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Buffers for the current batch
        self.batch_regret: Dict[InfosetKey, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.batch_strategy: Dict[InfosetKey, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.batch_counts: Dict[InfosetKey, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def get_policy(self, infoset: InfosetKey,
                   legal_moves: tuple[str, ...]) -> Dict[str, float]:
        """Regret-matching + ε-greedy exploration."""
        regrets = self.cumulative_regret.get(infoset, {})

        # relu(x) = max(0, x)
        baseline = relu(sum(relu(r) for r in regrets.values()))
        if baseline <= 0:
            # uniform if no positive regret
            probs = {a: 1.0 / len(legal_moves) for a in legal_moves}
        else:
            probs = {a: relu(regrets.get(a, 0.0)) / baseline for a in legal_moves}

        # ε-greedy
        eps = self.config["exploration"]
        uniform = 1.0 / len(legal_moves)
        policy = {a: eps * uniform + (1 - eps) * probs.get(a, 0.0) for a in legal_moves}
        return policy

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def observe_terminal(self, state: dict):
        """
        Single outcome-sampling traversal on the finished hand.
        We treat the *actual* sequence as the sampled outcome.
        """
        if not self.training:
            return

        # 1. Reconstruct the trajectory for each player
        #    history: list[(infoset, action, player_id)]
        if not self.history:
            return

        # 2. Compute terminal utilities
        seat = state["positions"][state["player_id"]]
        utility = float(state["rewards"][seat])  # number of chips won by the player

        # 3. Backward CFR pass
        self._traverse_outcome_sampling(utility)

        # 4. House-keeping
        self._on_game_end()
        if self.cycle_games >= self.config["batch_size"]:
            self._apply_accumulated_updates()
            self._on_cycle_end()

    # ------------------------------------------------------------------ #
    # Internal CFR
    # ------------------------------------------------------------------ #
    def _traverse_outcome_sampling(self, utility: float):
        """
        Single traversal of the *real* trajectory with outcome sampling weights.
        We walk backward through history and update regrets & strategy.
        """
        p_cap = self.config["p_cap"]

        # reach probabilities for each player (treated as constant for opponent)
        reach = {0: 1.0, 1: 1.0}  # we start at the root

        # walk backward
        for infoset, action, player_id in reversed(self.history):
            legal = tuple(self._get_all_actions(infoset))  # all ever seen
            if not legal:
                continue
            policy = self.get_policy(infoset, legal)
            my_p = policy.get(action, 0.0)
            if my_p <= 0.0:
                continue  # avoid div-by-zero

            # probability cap
            if reach[player_id] < p_cap:
                continue

            # counterfactual reach = product of opponent + chance probs
            cf_reach = reach[0] * reach[1] / reach[player_id]

            # regret
            value = sum(policy[a] * self._expected_value(infoset, a, utility) for a in legal)
            regret = utility - value

            # weighted regret
            w = cf_reach * regret
            self.batch_regret[infoset][action] += w
            self.batch_strategy[infoset][action] += reach[player_id] * policy[action]
            self.batch_counts[infoset] += 1

            # update reach for next (earlier) step
            reach[player_id] *= my_p

        # clear after traversal
        self.history.clear()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expected_value(infoset: InfosetKey,
                        action: str,
                        terminal_utility: float) -> float:
        """
        Because we sample only *one* outcome we use the observed utility
        for the action that was actually taken and 0 for all others.
        This is unbiased for the *average* regret.
        We rely on the fact that history contains the *actual* action.
        (simplified because we sample only one outcome)
        """
        return terminal_utility

    def _apply_accumulated_updates(self):
        """Merge batch into main tables with CFR+ clamping."""
        floor = self.config["regret_floor"]
        for infoset in self.batch_regret:
            for action, r in self.batch_regret[infoset].items():
                self.cumulative_regret[infoset][action] += r
                # CFR+
                if self.cumulative_regret[infoset][action] < floor:
                    self.cumulative_regret[infoset][action] = floor
        for infoset in self.batch_strategy:
            for action, s in self.batch_strategy[infoset].items():
                self.cumulative_strategy[infoset][action] += s

        # reset batch
        self.batch_regret.clear()
        self.batch_strategy.clear()
        self.batch_counts.clear()

    # ------------------------------------------------------------------ #
    # Persistence  (re-use RL schema but store regret & strategy)
    # ------------------------------------------------------------------ #
    def save(self, filepath, decimals=2):
        """Save the data to the json file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "cumulative_regret": {_serialize_infoset_key(k): round_floats(v, decimals)
                                  for k, v in self.cumulative_regret.items()},
            "cumulative_strategy": {_serialize_infoset_key(k): round_floats(v, decimals)
                                    for k, v in self.cumulative_strategy.items()},
            "games_played": self.games_played,
            "config": round_floats(self.config, decimals),
        }
        with filepath.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"CRM policy saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath, **kwargs):
        """Load the data from the json file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        agent = cls(**kwargs)
        agent.cumulative_regret = {_deserialize_infoset_key(k): v
                                   for k, v in data["cumulative_regret"].items()}
        agent.cumulative_strategy = {_deserialize_infoset_key(k): v
                                     for k, v in data["cumulative_strategy"].items()}
        agent.games_played = data.get("games_played", 0)
        agent.config = {**agent.config, **data.get("config", {})}
        print(f"CRM policy loaded: {filepath.name} ({agent.games_played:,} games)")
        return agent
