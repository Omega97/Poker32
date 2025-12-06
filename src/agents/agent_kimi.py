"""
Kimi agent v1

UCB-style bonus + max-baseline policy-gradient update.
No neural nets – still a pure logit table.
"""
import math
from src.agents.rl_agent import AgentRL


class AgentUCB(AgentRL):
    """
    Adds an exploration bonus sqrt(ln(total_visits) / action_visits)
    on top of the vanilla REINFORCE gradient.
    Uses the same max-baseline as AgentEquilibrium and keeps entropy
    regularisation to prevent collapse.
    """

    def _apply_accumulated_updates(self):
        lr = self.config["learning_rate"]
        mom = self.config.get("momentum", 0.9)
        ent_beta = self.config.get("entropy_bonus", 0.01)
        min_logit = -self.config.get("logit_range", 20.0)

        for infoset, reward_dict in list(self.accumulated.items()):
            if not reward_dict:
                continue

            # ---- 1. average reward per action ----
            counts = self.action_counts.get(infoset, {})
            avg = {a: reward_dict[a] / counts[a] for a in reward_dict if counts.get(a, 0) > 0}
            if not avg:
                continue

            # ---- 2. max baseline ----
            baseline = max(avg.values()) if max(avg.values()) > 0 else 0.0
            adv = {a: r - baseline for a, r in avg.items()}

            # ---- 3. UCB bonus ----
            total_visits = sum(counts.values())
            ucb = {}
            for a in avg:
                n = counts[a]
                bonus = math.sqrt(math.log(total_visits + 1) / (n + 1e-6))
                ucb[a] = bonus

            # ---- 4. entropy bonus (same as AgentEquilibrium) ----
            logits = self.logits.setdefault(infoset, {})
            legal = sorted(self._get_all_actions(infoset))
            action_logits = [logits.get(ac, 0.0) for ac in legal]
            max_l = max(action_logits)
            probs = self._softmax(action_logits, temp=1.0)
            entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
            ent_bonus = {ac: ent_beta * (math.log(probs[i] + 1e-12) + 1)
                         for i, ac in enumerate(legal)}

            # ---- 5. final gradient ----
            grad = {}
            for ac in legal:
                g = adv.get(ac, 0.0) + ucb.get(ac, 0.0) + ent_bonus.get(ac, 0.0)
                grad[ac] = g

            # ---- 6. L2 normalise & scale ----
            vec = [grad.get(ac, 0.0) for ac in legal]
            norm = math.hypot(*vec) or 1.0
            desired = lr * math.sqrt(len(legal))
            scaled = [g * desired / norm for g in vec]

            # ---- 7. momentum update + clip ----
            velocity = self.update_momentum.setdefault(infoset, {})
            for ac, delta in zip(legal, scaled):
                if ac not in logits:
                    logits[ac] = self.rng.uniform(-0.3, 0.3)
                v = velocity.get(ac, 0.0)
                velocity[ac] = mom * v + (1 - mom) * delta
                logits[ac] += velocity[ac]

            # max-normalise & hard clip
            if logits:
                max_l = max(logits.values())
                for ac in list(logits):
                    logits[ac] = max(logits[ac] - max_l, min_logit)

        # ---- 8. clean buffers ----
        self.accumulated.clear()
        self.action_counts.clear()

    # helper
    @staticmethod
    def _softmax(logits, temp=1.0):
        max_l = max(logits)
        exps = [math.exp((l - max_l) / temp) for l in logits]
        tot = sum(exps)
        return [e / tot for e in exps]
