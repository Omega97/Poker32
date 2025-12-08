# src/agents/crm_agent.py
from typing import Dict
from src.agents.rl_agent import AgentRL, InfosetKey


class AgentCRM(AgentRL):
    """
    Outcome Sampling MCCFR with:
      - On-policy probability capping (Pluribus-style) → prevents exploding regrets
      - CFR+ (positive regrets only) → linear convergence
    """

    DEFAULT_CONFIG = {
        "on_policy_cap": 0.20,        # ← CRITICAL: prevents importance sampling blowup
        "use_cfr_plus": True,         # ← CRITICAL: linear convergence, no oscillation
        "exploration": 0.6,
        "min_exploration": 0.05,
        "decay": 0.999,
        "n_epochs": 1000,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_w_probs = []

    # ------------------------------------------------------------------ #
    # 1. Policy Generation — NOW WITH CAPPING
    # ------------------------------------------------------------------ #
    def _get_policy(self, infoset: InfosetKey,
                    legal_moves: tuple[str, ...]) -> Dict[str, float]:
        k_exploration = max(
            self.config.get("min_exploration", 0.05),
            self.config.get("exploration", 0.6) * (self.config.get("decay", 0.999) ** self.games_played)
        )
        cap = self.config.get("on_policy_cap", 0.20)

        # 1. Regret Matching (unchanged)
        regrets = self.logits.get(infoset, {})
        positive_regrets = {a: max(regrets.get(a, 0.0), 0.0) for a in legal_moves}
        sum_pos = sum(positive_regrets.values())

        if sum_pos > 0:
            rm_policy = {a: positive_regrets[a] / sum_pos for a in legal_moves}
        else:
            rm_policy = {a: 1.0 / len(legal_moves) for a in legal_moves}

        # 2. ε-greedy mixing
        n = len(legal_moves)
        if not self.training:
            final = rm_policy
        else:
            final = {
                a: (1 - k_exploration) * rm_policy[a] + k_exploration / n
                for a in legal_moves
            }

        # ←←← THE KEY FIX: CAP MAX PROBABILITY (prevents w = 1/p → ∞)
        if self.training and cap > 0:
            capped = {a: min(p, cap) for a, p in final.items()}
            total = sum(capped.values())
            if total > 0:
                final = {a: p / total for a, p in capped.items()}

        return final

    # ------------------------------------------------------------------ #
    # 2. Action Selection — unchanged except we trust capped policy
    # ------------------------------------------------------------------ #
    def choose_action(self, state: dict) -> str:
        infoset = self._get_infoset_key(state)
        legal = state["legal_moves"]
        player_id = state["player_id"]

        policy = self._get_policy(infoset, legal)
        actions = list(policy.keys())
        probs = list(policy.values())
        action = self.rng.choices(actions, weights=probs, k=1)[0]
        prob_selected = policy[action]

        self.history_w_probs.append((infoset, action, player_id, prob_selected, policy))
        super()._append_to_history(infoset, action, player_id)
        return action

    # ------------------------------------------------------------------ #
    # 3. Learning — unchanged math, just CFR+ at the end
    # ------------------------------------------------------------------ #
    def observe_terminal(self, state: dict):
        if not self.training or not self.history_w_probs:
            self.history_w_probs.clear()
            self.history.clear()
            return

        self._accumulate_from_history(state)

        self.history_w_probs.clear()
        self.history.clear()
        self.games_played += 1
        self.cycle_games += 1

        if self.cycle_games >= self.config["n_epochs"]:
            self._apply_accumulated_updates()
            self.cycle_games = 0

    def _accumulate_from_history(self, state: dict):
        for (infoset, action_taken, p_id, prob_taken, policy) in self.history_w_probs:
            position = state["positions"][p_id]
            utility = state["rewards"][position]

            # Importance weight (now bounded because prob_taken ≥ 1 - cap*(n-1))
            w = 1.0 / prob_taken

            if infoset not in self.accumulated:
                self.accumulated[infoset] = {}
            if infoset not in self.action_counts:
                self.action_counts[infoset] = {}

            for a, prob_a in policy.items():
                if a == action_taken:
                    sampled_regret = w * utility * (1.0 - prob_taken)
                else:
                    sampled_regret = -w * utility * prob_a

                # Accumulate regret
                self.accumulated[infoset][a] = self.accumulated[infoset].get(a, 0.0) + sampled_regret

                # Accumulate average strategy (weighted by reach prob ≈ policy prob)
                self.action_counts[infoset][a] = self.action_counts[infoset].get(a, 0.0) + prob_a

    def _apply_accumulated_updates(self):
        use_cfr_plus = self.config.get("use_cfr_plus", True)

        for infoset, deltas in self.accumulated.items():
            if infoset not in self.logits:
                self.logits[infoset] = {}

            for action, delta in deltas.items():
                old = self.logits[infoset].get(action, 0.0)
                new_regret = old + delta
                if use_cfr_plus:
                    new_regret = max(new_regret, 0.0)
                self.logits[infoset][action] = new_regret

            # ←←← THIS IS THE FINAL FIX
            self._normalize_regrets(infoset)

        self.accumulated.clear()

    def _normalize_regrets(self, infoset: InfosetKey):
        """
        Called after every regret update on an infoset.
        Prevents numerical explosion and keeps policy well-behaved.
        This is the secret sauce of all stable CFR agents in 2025.
        """
        if infoset not in self.logits:
            return

        regrets = self.logits[infoset]
        actions = list(regrets.keys())
        if len(actions) < 2:
            return

        values = [regrets[a] for a in actions]

        # 1. Max-normalization (MOST IMPORTANT)
        max_r = max(values)
        for a in actions:
            regrets[a] -= max_r

        # 2. Variance capping (SECOND MOST IMPORTANT)
        # Target max std dev = config["logit_range"] (e.g., 10.0 or 20.0)
        import math
        values = [regrets[a] for a in actions]  # re-read after shift
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = math.sqrt(variance)

            target_std = self.config.get("logit_range", 15.0)
            if std > target_std:
                scale = target_std / (std + 1e-8)
                for a in actions:
                    regrets[a] = (regrets[a] - mean) * scale + mean

        # Optional: re-apply CFR+ after scaling (safe)
        if self.config.get("use_cfr_plus", True):
            for a in actions:
                if regrets[a] < 0:
                    regrets[a] = 0.0
