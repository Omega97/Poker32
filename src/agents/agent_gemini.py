"""Gemini agent v1"""
import math
from src.agents.rl_agent import AgentRL


class AgentMellow(AgentRL):
    """
    AgentMellow:
    1. Uses a Softmax-weighted (Mellow) baseline instead of Max-baseline.
       This ensures the 'best' action gets a positive advantage signal (Reward > Baseline),
       accelerating convergence compared to Max-baseline (Reward == Baseline).
    2. Adopts the 'Concentration Bonus' found in Grok/Kimi/Qwen (positive log probability),
       which encourages the agent to commit to the best strategy found rather than
       remaining eternally uncertain.
    """

    def _apply_accumulated_updates(self):
        # --- Configuration ---
        lr = self.config.get("learning_rate", 0.1)
        momentum = self.config.get("momentum", 0.9)
        # Note: Positive beta here encourages "Concentration" (Exploitation)
        entropy_beta = self.config.get("entropy_bonus", 0.01)
        logit_range = self.config.get("logit_range", 20.0)
        damping = self.config.get("damping", 0.99)
        # 'tau' controls the baseline softness.
        # tau=1.0 matches the sampling temperature.
        tau = self.config.get("temperature", 1.0)

        # --- Update Loop ---
        for infoset, reward_dict in list(self.accumulated.items()):
            if not reward_dict:
                continue

            # 1. Compute Average Rewards (Q-values) for visited actions
            counts = self.action_counts.get(infoset, {})
            avg_rewards = {
                a: reward_dict[a] / counts[a]
                for a in reward_dict
                if counts.get(a, 0) > 0
            }
            if not avg_rewards:
                continue

            # Get all known actions to ensure vector alignment
            actions = sorted(self._get_all_actions(infoset))

            # 2. Compute Mellow Baseline
            # Baseline = Sum(w_i * Q_i) / Sum(w_i), where w_i = exp(Q_i / tau)
            # We only use rewards observed *this cycle* to calculate the baseline
            # to prevent stale data from dragging the baseline down.
            valid_qs = [avg_rewards[a] for a in actions if a in avg_rewards]

            if not valid_qs:
                baseline = 0.0
            else:
                max_q = max(valid_qs)
                # Softmax weights (numerically stable)
                shift_qs = [(q - max_q) / tau for q in valid_qs]
                exps = [math.exp(sq) for sq in shift_qs]
                sum_exps = sum(exps)

                if sum_exps > 0:
                    weights = [e / sum_exps for e in exps]
                    baseline = sum(w * q for w, q in zip(weights, valid_qs))
                else:
                    baseline = max_q

            # 3. Compute Advantages
            # Adv(a) = Q(a) - Baseline
            # Note: Best action usually has Q > Baseline, getting a POSITIVE signal.
            advantages = {}
            for a in actions:
                # If action wasn't visited, we treat Q as 0.0 (neutral assumption)
                q = avg_rewards.get(a, 0.0)
                advantages[a] = q - baseline

            # 4. Compute Concentration Bonus (The "Winner's Term")
            # Calculate current policy probabilities
            logits = self.logits.setdefault(infoset, {})
            for a in actions:
                if a not in logits:
                    logits[a] = self.rng.uniform(-0.1, 0.1)

            action_logits = [logits[a] for a in actions]
            max_l = max(action_logits)
            e_logits = [math.exp(l - max_l) for l in action_logits]
            sum_e = sum(e_logits)
            probs = [e / sum_e for e in e_logits]

            # Bonus = beta * (log(p) + 1).
            # High prob -> Positive update. Low prob -> Negative update.
            # This pushes the agent towards a pure strategy (Nash Equilibrium).
            conc_bonus = {}
            for i, a in enumerate(actions):
                p = probs[i]
                log_p = math.log(p + 1e-12)
                conc_bonus[a] = entropy_beta * (log_p + 1.0)

            # 5. Form Gradient & Normalize
            grads = [advantages[a] + conc_bonus[a] for a in actions]

            # L2 Normalization scaled by sqrt(N) (Standard from Grok/Qwen)
            norm = math.hypot(*grads) or 1.0
            scale_factor = lr * math.sqrt(len(actions)) / norm
            scaled_grads = [g * scale_factor for g in grads]

            # 6. Momentum & Update
            velocity = self.update_momentum.setdefault(infoset, {})
            for i, a in enumerate(actions):
                delta = scaled_grads[i]

                # Momentum (Polyak)
                v_old = velocity.get(a, 0.0)
                v_new = momentum * v_old + (1.0 - momentum) * delta
                velocity[a] = v_new

                # Apply
                logits[a] += v_new

                # Damping & Clipping
                logits[a] *= damping
                logits[a] = max(min(logits[a], logit_range), -logit_range)

        # Cleanup
        self.accumulated.clear()
        self.action_counts.clear()
