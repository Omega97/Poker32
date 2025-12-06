"""Qwen agent v2"""
import math
from src.agents.rl_agent import AgentRL


class AgentAdvanced(AgentRL):
    """
    An advanced RL agent for Poker32.
    Uses max-baseline advantage updates, entropy regularization,
    and an adaptive learning rate inversely proportional to action counts.
    """
    def _apply_accumulated_updates(self):
        """
        Applies accumulated rewards using an advantage-based update rule
        with entropy regularization and an adaptive learning rate based on action counts.
        """
        base_lr = self.config["learning_rate"]
        momentum = self.config.get("momentum", 0.9)
        entropy_beta = self.config.get("entropy_bonus", 0.01)
        logit_range = self.config.get("logit_range", 20.0)
        init_range = self.config.get("init_range", 0.1)
        temperature = self.config.get("temperature", 1.0)

        for infoset, reward_dict in list(self.accumulated.items()):
            if not reward_dict:
                continue

            counts = self.action_counts.get(infoset, {})
            # Calculate average reward per action for this cycle
            avg_rewards = {
                a: reward_dict[a] / counts[a]
                for a in reward_dict
                if counts.get(a, 0) > 0
            }
            if not avg_rewards:
                continue

            # 1. Compute advantages using the MAX baseline
            baseline = max(avg_rewards.values())
            advantages = {a: r - baseline for a, r in avg_rewards.items()}

            # 2. Calculate current policy entropy and entropy bonus gradient
            logits = self.logits.setdefault(infoset, {})
            legal_actions = sorted(list(self._get_all_actions(infoset))) # Sort for consistency

            # Initialize new logits if needed
            for a in legal_actions:
                if a not in logits:
                    logits[a] = self.rng.uniform(-init_range, init_range)

            # Calculate current policy probabilities (pi) using logits and temperature
            action_logits = [logits.get(a, 0.0) for a in legal_actions]
            if temperature > 0:
                temp_scaled_logits = [l / temperature for l in action_logits]
                max_l = max(temp_scaled_logits)
                shifted = [l - max_l for l in temp_scaled_logits]
                exps = [math.exp(l) for l in shifted]
                total = sum(exps) or 1.0
                pi = [e / total for e in exps]
            else: # Deterministic policy if temperature is 0
                max_idx = action_logits.index(max(action_logits))
                pi = [0.0] * len(action_logits)
                pi[max_idx] = 1.0

            # Calculate entropy H = -sum(p * log(p))
            entropy = -sum(p * math.log(p + 1e-12) for p in pi if p > 0)
            # Calculate entropy bonus gradient: d/d_logits[ H ] = p * (log(p) + 1)
            entropy_bonus_grad = {
                a: entropy_beta * (math.log(pi[i] + 1e-12) + 1) * pi[i]
                for i, a in enumerate(legal_actions)
            }

            # 3. Prepare updates for each legal action
            updates = {}
            for a in legal_actions:
                # Advantage might be 0 if action wasn't taken this cycle
                adv = advantages.get(a, 0.0)
                ent_bonus = entropy_bonus_grad.get(a, 0.0)
                combined_grad = adv + ent_bonus

                # 4. Adaptive Learning Rate: scale base_lr by 1 / count(a) for this cycle
                # This makes frequent actions' logits change less drastically per update.
                # If an action wasn't taken this cycle (count=0), its update is based only on entropy.
                action_count = counts.get(a, 0)
                if action_count > 0:
                    adaptive_lr = base_lr / action_count
                    updates[a] = combined_grad * adaptive_lr
                else:
                    # Only entropy bonus contributes for actions not taken this cycle
                    updates[a] = ent_bonus * base_lr # Apply base LR here as no count-based scaling occurred for entropy

            # 5. Apply momentum and update logits
            velocity = self.update_momentum.setdefault(infoset, {})
            for a, update_val in updates.items():
                old_velocity = velocity.get(a, 0.0)
                # Momentum update: v = momentum * old_v + (1 - momentum) * new_update_val
                # Note: This is a slight variation from standard momentum where the new gradient is used.
                # Here, we apply momentum to the already calculated 'update_val'.
                # Standard momentum for the *value* is: new_val = old_val + v_{t+1}; v_{t+1} = beta * v_t + lr * grad
                # Our 'update_val' is like 'lr * grad'. So, v_{t+1} = beta * v_t + update_val
                new_velocity = momentum * old_velocity + update_val
                velocity[a] = new_velocity

                # Apply the velocity update to the logit
                logits[a] += new_velocity

            # 6. Apply damping and clipping to logits
            damping = self.config.get("damping", 0.99)
            for a in logits:
                logits[a] = logits[a] * damping
                logits[a] = max(min(logits[a], logit_range), -logit_range)

        # Clear accumulated rewards and counts for the next cycle
        self.accumulated.clear()
        self.action_counts.clear()
