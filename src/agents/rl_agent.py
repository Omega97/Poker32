"""
src/agents/rl_agent.py
Base AgentRL class
"""
import random
import math
import json
from pathlib import Path
from typing import Dict, Any
from src.agent import Agent, InfosetKey
from src.utils import round_floats


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
        "temperature": 1.0,
        "n_epochs": 5_000,
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
        self.history: list[tuple[InfosetKey, str, int]] = []

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

    def _get_policy(self, infoset: InfosetKey, legal_moves: tuple[str, ...]) -> Dict[str, float]:
        """
        Compute action probabilities via softmax over logits for a given infoset.

        If temperature is zero, returns a greedy (deterministic) policy.
        Otherwise, applies softmax with temperature scaling and log-sum-exp
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
        temp = self.config.get("temperature", 1.)
        logits = self.logits.get(infoset, {})
        action_logits = [logits.get(a, 0.0) for a in legal_moves]

        if temp == 0:  # greedy
            idx = max(range(len(action_logits)), key=action_logits.__getitem__)
            policy = {a: 0.0 for a in legal_moves}
            policy[legal_moves[idx]] = 1.0
            return policy

        # Softmax with log-sum-exp trick
        max_logit = max(action_logits)
        shifted = [x - max_logit for x in action_logits]
        exps = [math.exp(x / temp) for x in shifted]
        total = sum(exps)
        return {a: exp / total for a, exp in zip(legal_moves, exps)}

    def _append_to_history(self, infoset: InfosetKey, action: str, player_id: int):
        self.history.append((infoset, action, player_id))

    def choose_action(self, state: dict) -> str:
        infoset = self._get_infoset_key(state)
        legal = state["legal_moves"]
        player_id = state["player_id"]

        policy = self._get_policy(infoset, legal)

        # Sample action
        actions = list(policy.keys())
        probs = list(policy.values())
        action = self.rng.choices(actions, weights=probs, k=1)[0]

        # Record for later update
        self._append_to_history(infoset, action, player_id)

        if self.verbose:
            dct = {a: f'{p:.3f}' for a, p in policy.items()}
            print(f"Infoset {infoset} | Policy: {dct} → {action}")

        return action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def _accumulate_from_history(self, state):
        """Apply rewards from 'history' to 'accumulated'."""
        for i, (infoset, action, player_id) in enumerate(self.history):
            # Accumulate update: only the taken action gets updated
            if infoset not in self.accumulated:
                self.accumulated[infoset] = {}

            position = state["positions"][player_id]
            reward = state["rewards"][position]
            r0 = self.accumulated[infoset].get(action, 0)
            self.accumulated[infoset][action] = r0 + reward

            if infoset not in self.action_counts:
                self.action_counts[infoset] = {}
            c0 = self.action_counts[infoset].get(action, 0)
            self.action_counts[infoset][action] = c0 + 1

    def observe_terminal(self, state: dict):
        """
        Learn from the outcome of a completed hand by accumulating policy updates.

        For each action the agent took during the hand (stored in `self.history`),
        this method accumulates an update proportional to the player's reward:
            Δlogit = learning_rate * reward

        Updates are **not applied immediately**; they are stored in `self.accumulated`
        and applied only after every `n_epochs` games (at cycle boundaries) to reduce
        variance and improve convergence in self-play.

        This method is a no-op if `self.training` is False.

        Parameters
        ----------
        state : dict
            Terminal state containing:
              - 'rewards': tuple of (p0_reward, p1_reward)
              - 'position': this agent's player index
              - other metadata (unused here)
        """
        if not self.training or not self.history:
            return

        self._accumulate_from_history(state)

        self.history.clear()

        # End of game → count it
        self.games_played += 1
        self.cycle_games += 1

        # End of cycle → apply accumulated updates
        if self.cycle_games >= self.config["n_epochs"]:
            self._apply_accumulated_updates()
            self.cycle_games = 0  # reset

    def _get_all_actions(self, infoset: InfosetKey) -> set[str]:
        """Return all actions ever seen at this infoset from all containers."""
        return (
                set(self.logits.get(infoset, {}).keys()) |
                set(self.accumulated.get(infoset, {}).keys()) |
                set(self.update_momentum.get(infoset, {}).keys())
        )

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
                             n_actions: int,
                             epsilon=1e-12) -> list[float]:
        """L2 normalize so ||grad|| = lr * sqrt(n_actions)."""
        # center
        baseline = sum(grad_vec) / len(grad_vec)
        # baseline = max(baseline, 0)
        v = [g - baseline for g in grad_vec]

        # normalize and scale
        norm = math.hypot(*v)
        if norm < epsilon:
            return [0.0] * len(v)
        desired = self.config["learning_rate"] * math.sqrt(n_actions)
        return [g * (desired / norm) for g in v]

    def _apply_momentum_and_update(
            self,
            infoset: InfosetKey,
            actions: list[str],
            normalized_grad: list[float]
    ):
        """Final step: momentum, update logits, max-normalize, clip."""
        mom = self.config.get("momentum", 0.9)
        logit_range = self.config.get("logit_range", 10.0)
        init_range = self.config.get("init_range", 0.1)

        logits = self.logits.setdefault(infoset, {})
        velocity = self.update_momentum.setdefault(infoset, {})

        for action, ng in zip(actions, normalized_grad):
            # Lazy random init on first update
            if action not in logits:
                logits[action] = self.rng.uniform(- init_range, + init_range)

            # Momentum update
            v_old = velocity.get(action, 0.0)
            v_new = mom * v_old + (1.0 - mom) * ng
            velocity[action] = v_new

            # Apply to logit
            logits[action] += v_new

        # Cap the logits between '-logit_range' and +'logit_range'.
        if logits:
            for a in logits:
                # Damping effect
                damping = self.config.get("damping", 0.99)
                logits[a] *= damping

                # Clipping
                logits[a] = min(logits[a], logit_range)
                logits[a] = max(logits[a], -logit_range)

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

    def _apply_accumulated_updates(self):
        """Main update loop — now crystal clear and fully modular."""
        for infoset in list(self.accumulated.keys()):
            if infoset not in self.accumulated or not self.accumulated[infoset]:
                continue

            # Step 1: Compute average reward per action
            avg_rewards = self._compute_average_rewards(infoset)
            if not avg_rewards:
                continue

            # Step 2: Get full action list and build gradient vector
            actions = sorted(self._get_all_actions(infoset))
            grad_vec = [avg_rewards.get(a, 0.0) for a in actions]

            # Step 3: L2 normalize + scale by learning rate
            normalized_grad = self._normalize_and_scale(grad_vec, len(actions))
            if not any(normalized_grad):
                continue

            # Step 4: Apply momentum + update logits + max-norm + clip
            self._apply_momentum_and_update(infoset, actions, normalized_grad)

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
            _serialize_infoset_key(k): round_floats(v, decimals) for k, v in self.logits.items()
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

    def get_maturity(self, k=1.) -> float:
        """
        Measures how well a policy converged based on obvious mistakes.
        """
        if not self.logits:
            return 0.

        logits = []

        for (hand_char, history), strategy_dict in self.logits.items():

            if hand_char == "2":
                if "c" in strategy_dict and "f" in strategy_dict and "R" not in strategy_dict:
                    logits.append(-strategy_dict["c"])  # <- we want it to be negative
            # if hand_char == "A":
            #     if "f" in strategy_dict:
            #         logits.append(-strategy_dict["f"])  # <- we want it to be negative

        if len(logits):
            x = sum(logits) / len(logits) * k
            return math.tanh(max(0., x))
        else:
            return 0.


def load_rl_agent(
        filepath: Path | str,
        *,
        rng: random.Random | None = None,
        verbose: bool = False,
        training: bool = False,
) -> AgentRL:
    """ Load RL agent."""
    if rng is None:
        rng = random.Random()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    logits_raw = data["logits"]
    logits = {_deserialize_infoset_key(k): v for k, v in logits_raw.items()}
    games_played = data.get("games_played", 0)

    agent = AgentRL(rng=rng, verbose=False, training=training, name=filepath.stem)
    agent.logits = logits
    agent.games_played = games_played

    if verbose:
        if games_played < 10 ** 6:
            print(f"  Loaded {filepath.name:30} → {games_played / 10 ** 3:.0f}K games")
        else:
            print(f"  Loaded {filepath.name:30} → {games_played / 10 ** 6:.1f}M games")

    return agent
