import random
import pickle
import math
from pathlib import Path
from typing import Dict, Any
from src.agent import Agent, InfosetKey


class AgentRL(Agent):
    """
    Tabular self-play RL agent using additive logit updates.

    Policy: softmax over per-infoset logits.
    Training: after each hand, add ε × reward to the logit of every action taken.
    Updates are accumulated during a cycle (n_epochs games) and applied only at the end.
    This delayed update improves stability in alternating self-play.
    """

    DEFAULT_CONFIG = {
        "learning_rate": 0.1,      # ε in the update: logit += ε * reward
        "temperature": 0.5,        # softmax temperature for action selection
        "n_epochs": 100,        # games per cycle
        "n_cycles": 100,           # total training = n_epochs × n_cycles games
    }

    def __init__(
        self,
        rng: random.Random | None = None,
        config: Dict[str, Any] | None = None,
        policy_path: str | Path | None = None,
        training: bool = False,
        verbose: bool = False,
    ):
        super().__init__(rng=rng, verbose=verbose)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Logits: infoset → {action: logit}
        self.logits: Dict[InfosetKey, Dict[str, float]] = {}

        # Accumulated updates during current cycle (reset every n_epochs)
        self.accumulated: Dict[InfosetKey, Dict[str, float]] = {}

        self.games_played = 0
        self.cycle_games = 0

        if policy_path:
            self.load(policy_path)

        self.training = training

        # History of infosets visited this hand (for updates at terminal)
        self.history: list[tuple[InfosetKey, str]] = []  # (infoset, action_taken)

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

    def choose_action(self, state: dict) -> str:
        infoset = self._get_infoset_key(state)
        legal = state["legal_moves"]

        policy = self._get_policy(infoset, legal)

        # Sample action
        actions = list(policy.keys())
        probs = list(policy.values())
        action = self.rng.choices(actions, weights=probs, k=1)[0]

        # Record for later update
        self.history.append((infoset, action))

        if self.verbose:
            print(f"Infoset {infoset} | Policy: { {a: f'{p:.3f}' for a,p in policy.items()} } → {action}")

        return action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
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
        # Terminate if training mode is not ON
        if not self.training:
            return

        rewards = state["rewards"]  # tuple: reward for each player
        if not self.history:
            return

        lr = self.config["learning_rate"]
        for i, (infoset, action) in enumerate(self.history):
            # Accumulate update: only the taken action gets updated
            if infoset not in self.accumulated:
                self.accumulated[infoset] = {}

            player_id = i % 2
            reward = rewards[player_id]
            self.accumulated[infoset][action] = self.accumulated[infoset].get(action, 0.0) + lr * reward

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

    def _l2_normalize_and_scale(
            self,
            grad_vec: list[float],
            n_actions: int
    ) -> list[float]:
        """
        Normalize and scale a raw policy gradient vector to have controlled magnitude.

        The update vector is scaled so that its L2 norm equals `learning_rate * sqrt(n_actions)`.
        This stabilizes learning across infosets with different numbers of legal actions
        by preventing overly large updates in high-branching states.

        Parameters
        ----------
        grad_vec : list[float]
            Raw accumulated update values (one per action) at a given infoset.
        n_actions : int
            Number of legal actions at this infoset.

        Returns
        -------
        list[float]
            Scaled and normalized update vector with controlled magnitude.
        """
        import math
        lr = self.config["learning_rate"]
        norm = math.hypot(*grad_vec)
        if norm < 1e-12:
            return [0.0] * len(grad_vec)
        desired = lr * math.sqrt(n_actions)
        return [g * (desired / norm) for g in grad_vec]

    def _apply_momentum_and_update(
            self,
            infoset: InfosetKey,
            actions: list[str],
            normalized_grad: list[float],
            initial_logit_range=0.1,
    ):
        """
        Apply momentum-filtered updates to policy logits and enforce numerical stability.

        This method:
          - Integrates the normalized gradient into a velocity buffer using momentum,
          - Updates logits using the new velocity,
          - Initializes missing logits with small random values if needed,
          - Normalizes logits so the maximum is 0 (equivalent to softmax invariance),
          - Clips logits from below to avoid vanishing gradients (`min_logit` config).

        Parameters
        ----------
        infoset : tuple[str, str]
            The (hole_card, betting_branch) identifying the decision point.
        actions : list[str]
            Ordered list of actions corresponding to `normalized_grad`.
        normalized_grad : list[float]
            Gradient vector after L2 normalization and scaling.
        initial_logit_range : float, optional
            Range for uniform initialization of unseen action logits.

        Side Effects
        ------------
        Modifies `self.logits[infoset]` and `self.update_momentum[infoset]` in place.
        """
        gamma = self.config.get("momentum", 0.9)
        min_logit = -abs(self.config.get("min_logit", -20))
        logits = self.logits.setdefault(infoset, {})
        velocity = self.update_momentum.setdefault(infoset, {})

        # === 1. Momentum update (raw, unbounded) ===
        for action, ng in zip(actions, normalized_grad):
            old_v = velocity.get(action, 0.0)
            new_v = gamma * old_v + (1.0 - gamma) * ng
            velocity[action] = new_v

            old_logit = logits.get(action, self.rng.uniform(-initial_logit_range, 0.))
            logits[action] = old_logit + new_v

        # === 2. Max-normalize: subtract the current maximum ===
        if logits:
            max_logit = max(logits.values())
            for a in logits:
                logits[a] -= max_logit

        # === 3. Clip bottom at -20 (prevents numerical death) ===
        for a in logits:
            if logits[a] < min_logit:
                logits[a] = min_logit

    def _cleanup_empty_infoset(self, infoset: InfosetKey):
        """Remove infoset completely if it has no logits left."""
        if infoset not in self.logits or not self.logits[infoset]:
            self.logits.pop(infoset, None)
            self.update_momentum.pop(infoset, None)

    def _apply_accumulated_updates(self, initial_logit_range=0.1):
        """
        Apply delayed policy updates after each training cycle to improve stability.

        This method processes the accumulated gradient-like updates collected during
        `n_epochs` self-play games and applies them to the policy logits using:
          1. L2 normalization and scaling of the raw updates,
          2. Momentum-based velocity integration,
          3. Max-normalization of logits (so the best action has logit = 0),
          4. Lower clipping to prevent logits from vanishing numerically.

        Infers the full action support at each infoset by merging keys from logits,
        accumulated updates, and momentum buffers. Empty infosets are cleaned up.

        Parameters
        ----------
        initial_logit_range : float, optional
            Range for initializing logits of previously unseen actions (default: 0.1).
            Used only if an action appears in updates but not in current logits.
        """
        for infoset in list(self.accumulated.keys()):
            raw_updates = self.accumulated[infoset]
            if not raw_updates:
                continue

            actions = sorted(self._get_all_actions(infoset))
            if not actions:
                continue

            grad_vec = [raw_updates.get(a, 0.0) for a in actions]
            normalized = self._l2_normalize_and_scale(grad_vec, len(actions))

            has_update = self._apply_momentum_and_update(infoset, actions, normalized,
                                                         initial_logit_range=initial_logit_range)

            if not has_update:
                self._cleanup_empty_infoset(infoset)

        self.accumulated.clear()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

        data = {
            "logits": self.logits,
            "games_played": self.games_played,
            "config": self.config,
        }

        with filepath.open("wb") as f:
            pickle.dump(data, f)

        print(f"\nPolicy saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: str | Path, **kwargs):
        """
        Load a saved policy and return a fully configured AgentRL instance.
        Any config values passed in kwargs will OVERRIDE the saved ones.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Policy file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # === 1. Load saved data ===
        saved_logits = data["logits"]
        saved_games_played = data.get("games_played", 0)
        saved_config = data.get("config", {})

        # === 2. Merge config: kwargs wins over saved config ===
        final_config = {**saved_config, **kwargs.get("config", {})}

        # === 3. Create agent with correct training mode and RNG ===
        agent = cls(
            rng=kwargs.get("rng"),  # fresh or seeded RNG
            verbose=kwargs.get("verbose", False),
            config=final_config,
            policy_path=None,  # prevent recursive load
            training=kwargs.get("training", False),  # default: frozen for evaluation
        )

        # === 4. Restore learned parameters ===
        agent.logits = saved_logits
        agent.update_momentum = data.get("update_momentum", {})
        agent.games_played = saved_games_played

        # === 5. FINAL SAFETY: force config override (in case someone forgets) ===
        if "config" in kwargs:
            agent.config = final_config

        print(f"Policy loaded: {filepath.name}")
        print(f"  • Trained for {agent.games_played:,} games")
        print(f"  • Learning rate: {agent.config['learning_rate']}")
        print(f"  • Training mode: {agent.training}")

        return agent

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def reset_history(self):
        self.history.clear()


def load_rl_agent(
        filepath: Path | str,
        *,
        verbose: bool = False,
        training: bool = False,
) -> AgentRL:
    """
    Load an agent from disk.
    """
    filepath = Path(filepath)

    # Normal load path
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    # Agent must be NOT verbose in the tournament
    agent = AgentRL(rng=random.Random(), verbose=False, training=training)
    agent.logits = data["logits"]
    agent.update_momentum = data.get("update_momentum", {})
    agent.games_played = data.get("games_played", 0)

    if verbose:
        print(f"  Loaded {filepath.name:30} → {agent.games_played / 10**6:.1f}M games")

    return agent
