"""ChatGPT agent v1"""
import math
from src.agents.rl_agent import AgentRL
from src.poker32 import RANK_TO_IDX, N_RANKS


class AgentRefined(AgentRL):
    """
    Refined tabular policy updater.

    Key features / inspirations:
      - Max-baseline advantage (as in AgentEquilibrium) for stable policy gradients.
      - Entropy regularization to avoid premature collapse.
      - Per-infoset *hand-strength bias* (intrinsic prior) that discourages
        risky calls with very weak hole cards (addresses the '2 should never call' failure).
      - Adaptive per-infoset penalty on the 'call' action for weak cards (soft constraint).
      - L2-normalization of gradient + momentum + damping + clipping (familiar mechanics).
      - All external hyperparameters have defaults inside this method.
    """

    def _apply_accumulated_updates(self):
        # ------------- Defaults (can be overridden in self.config) -------------
        lr = self.config.get("learning_rate", 0.08)
        momentum = self.config.get("momentum", 0.90)
        entropy_beta = self.config.get("entropy_bonus", 0.012)
        logit_range = self.config.get("logit_range", 20.0)
        init_range = self.config.get("init_range", 0.12)
        damping = self.config.get("damping", 0.99)
        min_logit = -self.config.get("logit_range", 20.0)
        # Hand-strength related defaults
        hand_bias_scale = self.config.get("hand_bias_scale", 1.8)   # how strongly hand strength shifts baseline
        call_penalty_base = self.config.get("call_penalty_base", 1.5)  # base penalty on 'call' for weak hands
        call_penalty_exponent = self.config.get("call_penalty_exponent", 2.0)  # non-linear scaling
        force_zero_call_for = set(self.config.get("force_zero_call_for", ["2"]))  # optionally force certain cards
        eps = 1e-12

        # Iterate all infosets that gathered signals this cycle
        for infoset, reward_dict in list(self.accumulated.items()):
            if not reward_dict:
                continue

            counts = self.action_counts.get(infoset, {})
            # Average rewards per-action for this cycle
            avg_rewards = {
                a: reward_dict[a] / counts[a]
                for a in reward_dict
                if counts.get(a, 0) > 0
            }
            if not avg_rewards:
                continue

            # Actions considered at this infoset
            logits = self.logits.setdefault(infoset, {})
            legal = sorted(self._get_all_actions(infoset))

            # Ensure immortal actions initialized lazily
            for a in legal:
                if a not in logits:
                    logits[a] = self.rng.uniform(-init_range, +init_range)

            # --- Hand strength prior (based on infoset's hole card) ---
            # infoset is (hole_card, branch)
            hole = infoset[0] if isinstance(infoset, tuple) and len(infoset) >= 1 else None
            # default neutral strength
            if hole and hole in RANK_TO_IDX:
                # higher numeric strength -> stronger hand (A strongest -> idx 0)
                idx = RANK_TO_IDX[hole]
                strength = (N_RANKS - idx) / N_RANKS  # in (0,1], A close to 1, '2' close to 1/N_RANKS
                # normalize to [0,1]
                # strength already mapped A -> near 1, 2 -> near 1/N_RANKS
            else:
                strength = 0.5

            # Compute baseline: max(avg_rewards) with non-negative floor, then adjust by hand-strength bias
            baseline = max(max(avg_rewards.values()), 0.0)
            # Shift baseline downward for weak hands (so advantages for raises must overcome lower baseline less)
            # We subtract a bias proportional to (0.5 - strength) so weak hands (strength < 0.5) get negative shift
            hand_bias = hand_bias_scale * (0.5 - strength)
            baseline = baseline - hand_bias

            # --- Entropy precomputation (current policy) ---
            action_logits = [logits[a] for a in legal]
            max_l = max(action_logits)
            shifted = [l - max_l for l in action_logits]
            exps = [math.exp(l) for l in shifted]
            total = sum(exps) or 1.0
            probs = {a: e / total for a, e in zip(legal, exps)}

            # --- Prepare advantage + entropy bonus per action ---
            advantages = {a: (avg_rewards.get(a, 0.0) - baseline) for a in legal}

            # Entropy term (per-action): proportional to (log p + 1) as in AgentEquilibrium
            entropy = -sum(p * math.log(p + eps) for p in probs.values())
            entropy_bonus = {a: entropy_beta * (math.log(probs[a] + eps) + 1.0) for a in legal}

            # --- Call-action penalty for weak hands (softly discourages calling with weak cards) ---
            # Penalty scales non-linearly with weakness: weaker -> bigger penalty
            # penalty = call_penalty_base * (max(0, 0.5 - strength))**exponent
            weakness = max(0.0, 0.5 - strength)
            call_penalty = call_penalty_base * (weakness ** call_penalty_exponent)

            # If explicitly forcing certain cards never to call (e.g., '2'), apply a very large penalty
            force_no_call = (hole in force_zero_call_for)

            # --- Build gradient vector combining advantage, entropy, and exploration/penalty ---
            grad = {}
            for a in legal:
                adv = advantages.get(a, 0.0)
                ent = entropy_bonus.get(a, 0.0)
                g = adv + ent

                # Penalize 'c' (call) if hand is weak
                if a == 'c':
                    if force_no_call:
                        # Very strong hard penalty (effectively zero probability)
                        g -= max(100.0, logit_range * 10.0)
                    else:
                        g -= call_penalty

                grad[a] = g

            # --- L2 normalize + scale to learning budget ---
            vec = [grad[a] for a in legal]
            # center by mean to remove uniform shift
            mean_vec = sum(vec) / len(vec)
            centered = [v - mean_vec for v in vec]
            norm = math.hypot(*centered) or 1.0
            scale = lr * math.sqrt(len(legal)) / norm
            normalized = [v * scale for v in centered]

            # --- Momentum + update + ensure immortality for unseen actions ---
            velocity = self.update_momentum.setdefault(infoset, {})
            for a, delta in zip(legal, normalized):
                if a not in logits:
                    logits[a] = self.rng.uniform(-init_range, +init_range)
                v_old = velocity.get(a, 0.0)
                v_new = momentum * v_old + (1.0 - momentum) * delta
                velocity[a] = v_new
                logits[a] += v_new

            # --- Max-normalize + floor at min_logit (keeps diversity and avoids drift) ---
            if logits:
                max_after = max(logits.values())
                for a in list(logits.keys()):
                    logits[a] = max(logits[a] - max_after, min_logit)

            # --- Damping + clipping ---
            for a in legal:
                logits[a] *= damping
                if logits[a] > logit_range:
                    logits[a] = logit_range
                elif logits[a] < -logit_range:
                    logits[a] = -logit_range

        # --- Clean / decay accumulators similarly to AgentRL ---
        gamma = self.config.get("momentum", 0.0)
        if gamma <= 0:
            self.accumulated.clear()
            self.action_counts.clear()
        else:
            self.accumulated = {
                infoset: {k: v * gamma for k, v in d.items()}
                for infoset, d in self.accumulated.items()
            }
            self.action_counts = {
                infoset: {k: v * gamma for k, v in d.items()}
                for infoset, d in self.action_counts.items()
            }
