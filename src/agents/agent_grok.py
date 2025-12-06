"""Grok agent v1"""
import math
from src.agents.rl_agent import AgentRL


class AgentEquilibrium(AgentRL):
    """
    The final evolution.
    - Max-baseline advantage (the correct one)
    - Entropy regularization (prevents premature collapse)
    - Momentum + damping + proper L2 scaling
    - Immortal actions (never deleted)
    - Automatic name from filename
    """

    def _apply_accumulated_updates(self):
        lr = self.config["learning_rate"]
        momentum = self.config.get("momentum", 0.9)
        entropy_beta = self.config.get("entropy_bonus", 0.01)
        min_logit = -self.config.get("logit_range", 20.0)

        for infoset, reward_dict in list(self.accumulated.items()):
            if not reward_dict:
                continue

            # === 1. Average rewards + counts ===
            counts = self.action_counts.get(infoset, {})
            avg_rewards = {
                a: reward_dict[a] / counts[a]
                for a in reward_dict
                if counts.get(a, 0) > 0
            }
            if not avg_rewards:
                continue

            # === 2. MAX BASELINE (this is the truth) ===
            baseline = max(avg_rewards.values())
            if baseline < 0:
                baseline = 0.0
            advantages = {a: r - baseline for a, r in avg_rewards.items()}

            # === 3. Entropy bonus ===
            logits = self.logits.setdefault(infoset, {})
            legal = self._get_all_actions(infoset)
            action_logits = [logits.get(a, 0.0) for a in legal]
            max_l = max(action_logits)
            shifted = [l - max_l for l in action_logits]
            exps = [math.exp(l) for l in shifted]
            total = sum(exps) or 1.0
            probs = [e / total for e in exps]

            entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
            entropy_bonus = {a: entropy_beta * (math.log(probs[i] + 1e-12) + 1) for i, a in enumerate(legal)}

            # === 4. Final gradient ===
            grad = {}
            for a in legal:
                adv = advantages.get(a, 0.0)
                ent = entropy_bonus.get(a, 0.0)
                grad[a] = adv + ent  # ← both terms

            # === 5. L2 normalize + scale ===
            vec = [grad.get(a, 0.0) for a in legal]
            norm = math.hypot(*vec) or 1.0
            scale = lr * math.sqrt(len(legal)) / norm
            normalized = [g * scale for g in vec]

            # === 6. Momentum + update + clip (never delete) ===
            velocity = self.update_momentum.setdefault(infoset, {})
            for a, delta in zip(legal, normalized):
                if a not in logits:
                    logits[a] = self.rng.uniform(-0.3, 0.3)
                v = velocity.get(a, 0.0)
                velocity[a] = momentum * v + (1 - momentum) * delta
                logits[a] += velocity[a]

            # Max-normalize + clip
            if logits:
                max_l = max(logits.values())
                for a in list(logits):
                    logits[a] = max(logits[a] - max_l, min_logit)

        # Clear
        self.accumulated.clear()
        self.action_counts.clear()
