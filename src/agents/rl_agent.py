"""
src/agents/rl_agent.py
Base AgentRL class

# todo make perfectly deterministic
# todo remove serialization
"""
import random
import math
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
from src.agent import Agent, InfosetKey
from src.utils import round_floats, softmax, maturity, weighted_avg, normalize


_TEST_INFOSETS = {
        ("2", "RRRR"): "c",
        ("2", "RRD"): "c",
        ("2", "RDR"): "c",
        ("2", "RT"): "c",
        ("2", "DD"): "c",
        ("2", "DRR"): "c",
        ("2", "TR"): "c",
        ("2", "Q"): "c",
        ("2", "cRRRR"): "c",
        ("2", "cRRD"): "c",
        ("2", "cRDR"): "c",
        ("2", "cDRR"): "c",
        ("2", "cDD"): "c",
        ("2", "cTR"): "c",
        ("2", "cQ"): "c",
    }


def _serialize_infoset_key(key: InfosetKey) -> str:
    """Convert (hole, branch) → 'hole|branch'"""
    return f"{key[0]}|{key[1]}"


def _deserialize_infoset_key(s: str) -> InfosetKey:
    """Convert 'hole|branch' → (hole, branch)"""
    parts = s.split('|', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid serialized infoset key: {s}")
    return parts[0], parts[1]


class AgentRL(Agent):
    """
    Explicit policy for every decision-point of the game.
    """
    DEFAULT_CONFIG = {
        "learning_rate": 0.1,
        "batch_size": 5_000,
        "n_cycles": 50,
    }

    def __init__(
            self,
            rng: random.Random | None = None,
            config: Dict[str, Any] | None = None,
            policy_path: str | Path | None = None,
            name: str | None = None,
            training: bool = False,
            verbose: bool = False,
    ):
        super().__init__(rng=rng, verbose=verbose)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        self.logits: Dict[InfosetKey, Dict[str, float]] = {}
        self.accumulated: Dict[InfosetKey, Dict[str, float]] = {}
        self.games_played = 0
        self.cycle_games = 0
        self.name = name
        self.training = training
        self.action_counts: Dict[InfosetKey, Dict[str, int]] = {}
        self.history: list[tuple[InfosetKey, str]] = []
        self.update_momentum: Dict[InfosetKey, Dict[str, float]] = {}

        if policy_path:
            self.load(policy_path)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_infoset_key(state: dict) -> InfosetKey:
        """
        Convert a subjective game state into a unique infoset identifier.
        (hole_card, betting_branch)
        """
        return state["hole"], state["branch"]

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #
    def get_logits(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """ Get the logits dict for a given infoset. """
        return self.logits

    def set_logits(self, infoset: InfosetKey, logits_dict: Dict[str, float]):
        self.logits[infoset] = logits_dict

    def get_accumulated_reward(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        return self.accumulated

    def set_accumulated_reward(self, infoset: InfosetKey, action: str, value: float):
        self.accumulated[infoset][action] = value

    def add_accumulated_reward(self, infoset: InfosetKey, action: str, value: float):
        """Add 'value' to accumulated rewords-"""
        r0 = self.get_accumulated_reward()[infoset].get(action, 0)
        self.set_accumulated_reward(infoset, action, value=r0 + value)

    def add_action_counts(self, infoset: InfosetKey, action: str, n=1):
        """
        Add +1 to action_counts.
        Create dict of actions, counts if necessary.
        """
        if infoset not in self.action_counts:
            self.action_counts[infoset] = {}
        c0 = self.action_counts[infoset].get(action, 0)
        self.action_counts[infoset][action] = c0 + n

    def get_policy(self,
                   infoset: InfosetKey,
                   legal_moves: tuple[str, ...]) -> Dict[str, float]:
        """
        Compute action probabilities via softmax over logits for a given infoset.

        Otherwise, applies softmax with log-sum-exp
        for numerical stability. Unseen actions default to logit = 0.

        Parameters
        ----------
        infoset : tuple[str, str]
            The (hole_card, betting_branch) defining the subjective state.
        legal_moves : tuple[str, ...]
            Actions allowed by the game rules in this state.
        Returns
        -------
        dict[str, float]
            Probability distribution over legal actions (sums to 1).
        """
        spot_logits = self.get_logits().get(infoset, dict())
        action_logits = [spot_logits.get(a, 0.) for a in legal_moves]
        return softmax(legal_moves, action_logits)

    def _get_all_actions(self, infoset: InfosetKey) -> set[str]:
        """Return all actions ever seen at this infoset from all containers."""
        return (
                set(self.get_logits().get(infoset, dict()).keys()) |
                set(self.get_accumulated_reward().get(infoset, {}).keys()) |
                set(self.update_momentum.get(infoset, {}).keys())
        )

    def get_action_counts(self, infoset: InfosetKey, action: str) -> int:
        """Get number of action counts in that spot for that action."""
        counts = self.action_counts.get(infoset, {})
        n = counts.get(action, 0)
        return n

    def get_maturity(self) -> float:
        """
        Measures how well a policy converged based on obvious mistakes.
        """
        if not self.get_logits():
            return 0.

        logits = []
        for (hand_char, history), strategy_dict in self.get_logits().items():
            if hand_char == "2":
                if "c" in strategy_dict and "f" in strategy_dict and "R" not in strategy_dict:
                    logits.append(-strategy_dict["c"])  # <- we want it to be negative
            elif hand_char == "A":
                if "f" in strategy_dict:
                    logits.append(-strategy_dict["f"])  # <- we want it to be negative
                if "c" in strategy_dict and "R" in strategy_dict:
                    logits.append(-strategy_dict["c"])  # <- we want it to be negative

        return maturity(logits)

    def append_to_history(self, infoset: InfosetKey, action: str):
        """Append state info to history."""
        self.history.append((infoset, action))

    def clear_history(self):
        self.history.clear()

    # ------------------------------------------------------------------ #
    # Action Selection
    # ------------------------------------------------------------------ #
    def choose_action(self, state: dict) -> str:
        """
        Choose one of the legal actions, according to the policy.
        Record the choice in the history.
        """
        infoset = self._get_infoset_key(state)
        legal = state["legal_moves"]
        policy = self.get_policy(infoset, legal)

        # Sample action
        actions = list(policy.keys())
        probs = list(policy.values())
        action = self.rng.choices(actions, weights=probs, k=1)[0]

        # Record for later update
        if self.training:
            self.append_to_history(infoset, action)

        return action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def _accumulate_from_history(self, state):
        """
        Scan the history, and apply the reward of this state to
        the accumulated rewards of the trajectory, then clear the history.
        """
        for infoset, action in self.history:

            # Accumulate update: only the taken action gets updated
            if infoset not in self.accumulated:
                self.accumulated[infoset] = {}

            # Get reward from state; the reward for the player that performed the action
            hole_card, strand = infoset
            position_ids = state["position_ids"]  # Dict[str, int]
            rewards = state["rewards"]  # Tuple[int, ...]
            position = ("SB", "BB")[len(strand) % 2]
            id_player_performed_action = position_ids[position]
            reward = rewards[id_player_performed_action]

            # Update rewards and visit counts
            self.add_accumulated_reward(infoset, action, value=reward)
            self.add_action_counts(infoset, action)

            if self.verbose:
                print(f'> accumulating: {position} {reward}')

        # Clear the history
        self.clear_history()

    # ------------------------------------------------------------------ #
    # After-game procedures
    # ------------------------------------------------------------------ #
    def observe_terminal(self, state: dict):
        """
        After a game ends, learn from the outcome of a completed
        hand by accumulating policy updates.

        For each action the agent took during the hand (stored in `self.history`),
        this method accumulates an update proportional to the player's reward:
            Δlogit = learning_rate * reward

        Updates are **not applied immediately**; they are stored in `self.accumulated`
        and applied only after every `batch_size` games (at cycle boundaries) to reduce
        variance and improve convergence in self-play.

        This method is a no-op if `self.training` is False.

        Parameters
        ----------
        state : dict
            Terminal state containing:
              - 'player_id': int (0 or 1)
              - 'reward': int
              - 'position': str (BB or SB)
              - 'positions': dict of int, str; for example: {0: 'BB', 1: 'SB'}
              - 'names': list of player names,
              - 'hole_cards': for example: ['A', 'T']
              - 'branch': for example: 'Qf'
              - 'rewards': tuple of int, one for each player_id; for example: (2, -2)
        """
        if self.verbose:
            print(f"> observing: {state}")
            print(f"> branch: {state['branch']}")
            print(f"> history: {self.history}")

        if not self.training or not self.history:
            return

        # Apply rewards from self.history to self.accumulated
        self._accumulate_from_history(state)

        # Book-keeping
        self._on_game_end()

    def _on_game_end(self):
        """Update game counts."""
        self.games_played += 1
        self.cycle_games += 1

    def _on_cycle_end(self):
        self.cycle_games = 0

    def update_parameters(self):
        """
        After a batch of games is completed (end of cycle), update the
        parameters of the model.
        """
        self._apply_accumulated_updates()
        self._on_cycle_end()

    def _compute_average_rewards(self, infoset: InfosetKey) -> Dict[str, float]:
        """Return {action: average_reward} for actions seen this cycle."""
        avg = {}
        total_samples = 0
        counts = self.action_counts.get(infoset, {})
        rewards = self.accumulated.get(infoset, {})

        for action in rewards:
            count = counts.get(action, 0)
            if count > 0:
                avg[action] = rewards[action] / count
                total_samples += count

        return avg if total_samples > 0 else {}

    def _normalize_and_scale(self, grad_vec: list[float],
                             action_counts: List[int]
                             ) -> list[float]:
        """L2 normalize so ||grad|| = lr * sqrt(n_actions)."""
        n_actions = len(action_counts)

        # center
        baseline = weighted_avg(grad_vec, action_counts)
        v = [grad_vec[i] - baseline if action_counts[i] else 0. for i in range(n_actions)]

        # scale
        length = self.config["learning_rate"] * math.sqrt(n_actions)
        v = normalize(v, length)

        return v


    def _apply_momentum_and_update(
            self,
            infoset: InfosetKey,
            actions: list[str],
            normalized_grad: list[float]
    ):
        """
        Final step: momentum, update logits, max-normalize, clip.
        """
        mom = self.config.get("momentum", 0.9)
        logit_range = self.config.get("logit_range", 10.0)
        init_range = self.config.get("init_range", 0.1)
        damping = self.config.get("damping", 0.99)

        spot_logits = self.logits.setdefault(infoset, {})
        velocity = self.update_momentum.setdefault(infoset, {})

        for action, ng in zip(actions, normalized_grad):
            # Lazy random init on first update
            if action not in spot_logits:
                spot_logits[action] = self.rng.uniform(-init_range, +0)

            # Momentum update
            v_old = velocity.get(action, 0.0)
            v_new = mom * v_old + (1.0 - mom) * ng
            velocity[action] = v_new

            # Apply to logit
            spot_logits[action] += v_new

        # Cap the logits between '-logit_range' and +'logit_range'.
        if spot_logits:
            for i, a in enumerate(spot_logits):
                # Only apply on modified logits
                if normalized_grad[i] != 0.:
                    # Damping effect
                    spot_logits[a] *= damping

                    # Clipping
                    spot_logits[a] = min(spot_logits[a], +logit_range)
                    spot_logits[a] = max(spot_logits[a], -logit_range)

    def _clear_accumulated_and_counts(self):
        """
        Clear rewards and counts.
        If momentum is > 0, then apply gradual decay instead.
        """
        gamma = self.config.get("momentum", 0.)
        if gamma <= 0:
            self.accumulated.clear()
            self.action_counts.clear()
        else:
            self.accumulated = {infoset: {k: v * gamma for k, v in d.items()}
                                for infoset, d in self.accumulated.items()}
            self.action_counts = {infoset: {k: v * gamma for k, v in d.items()}
                                  for infoset, d in self.action_counts.items()}

    def _sanity_check(self):
        for infoset, action in _TEST_INFOSETS.items():
            value = self.get_logits().get(infoset, dict()).get(action, None)
            if value is not None:
                assert value <= 0, f'{action}|{infoset} -> {value:.4f} > 0\n'

    def _apply_accumulated_updates(self):
        """
        Update logits based on the average accumulated update.
        Used in 'self.update_parameters'.
        """
        # Sort for repeatability
        infosets = sorted(list(self.accumulated.keys()))

        for infoset in infosets:
            if infoset not in self.accumulated or not self.accumulated[infoset]:
                continue

            # Step 1: Compute average reward per action
            avg_rewards = self._compute_average_rewards(infoset)
            if not avg_rewards:
                continue

            # Step 2: Get full action list and build gradient vector
            actions = sorted(self._get_all_actions(infoset))
            grad_vec = [avg_rewards.get(a, 0.0) for a in actions]
            action_counts = [self.get_action_counts(infoset, a) for a in actions]

            # Step 3: L2 normalize + scale by learning rate
            normalized_grad = self._normalize_and_scale(grad_vec, action_counts)
            if not any(normalized_grad):
                continue

            # Step 4: Apply momentum + update logits + max-norm + clip
            self._apply_momentum_and_update(infoset, actions, normalized_grad)

        self._sanity_check()

        # Clean up
        self._clear_accumulated_and_counts()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, filepath: str | Path, decimals=2):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert tuple keys to strings
        serializable_logits = {
            _serialize_infoset_key(k): round_floats(v, decimals) for k, v in self.get_logits().items()
        }

        data = {
            "logits": serializable_logits,
            "games_played": self.games_played,
            "config": round_floats(self.config, decimals),
        }

        with filepath.open("w") as f:
            json.dump(data, f, indent=2)

        print(f"Policy saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: str | Path, **kwargs):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Policy file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        saved_logits_raw = data["logits"]
        saved_games_played = data.get("games_played", 0)
        saved_config = data.get("config", {})

        # Deserialize keys
        saved_logits = {
            _deserialize_infoset_key(k): v for k, v in saved_logits_raw.items()
        }

        final_config = {**saved_config, **kwargs.get("config", {})}

        agent = cls(
            rng=kwargs.get("rng"),
            verbose=kwargs.get("verbose", False),
            config=final_config,
            policy_path=None,
            name=kwargs.get("name", None),
            training=kwargs.get("training", False),
        )

        agent.logits = saved_logits
        # Note: update_momentum is not saved in JSON version (optional: add if needed)
        agent.games_played = saved_games_played

        if "config" in kwargs:
            agent.config = final_config

        print(f"\nPolicy loaded: {filepath.name}")
        print(f"  • Trained for {agent.games_played:,} games")
        print(f"  • Learning rate: {agent.config['learning_rate']}")
        print(f"  • Training mode: {agent.training}")

        return agent

    @staticmethod
    def _advantage_vector(rewards: list[float]) -> list[float]:
        """
        Return n-length vector whose largest entry is +(n-1) and all others -1.
        Sum is zero ⇒ no bias, variance is minimised for uniform policy.
        """
        n = len(rewards)
        best_idx = max(range(n), key=rewards.__getitem__)
        return [n - 1 if i == best_idx else -1 for i in range(n)]


def load_rl_agent(
        filepath: Path | str,
        *,
        rng: random.Random | None = None,
        name: str = None,
        verbose: bool = False,
        training: bool = False,
) -> AgentRL:
    """ Load RL agent."""
    if rng is None:
        rng = random.Random()
    if name is None:
        name = filepath.stem

    # Load the data from file
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    logits_raw = data["logits"]
    logits = {_deserialize_infoset_key(k): v for k, v in logits_raw.items()}
    games_played = data.get("games_played", 0)

    # Create agent instance
    agent = AgentRL(rng=rng, verbose=verbose, training=training, name=name)
    agent.logits = logits
    agent.games_played = games_played

    if games_played < 10 ** 6:
        print(f"  Loaded {filepath.name:30} → {games_played / 10 ** 3:.0f}K games")
    else:
        print(f"  Loaded {filepath.name:30} → {games_played / 10 ** 6:.1f}M games")

    return agent
