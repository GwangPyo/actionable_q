from stable_baselines3.sac import SAC
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update


class ActionableSAC(SAC):

    def offline_train_step(self, replay_data):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        regularization_losses = []
        actor_losses, critic_losses = [], []

        # relabel rewards
        with th.no_grad():
            qf_reward, _ = th.min(th.cat(self.critic_target(replay_data.observations, replay_data.actions), dim=1),
                                  dim=1, keepdim=True)
        # set 1 for goal reached, otherwise if rando goal, set qf
        # otherwise all zero
        rewards = (1 - replay_data.rewards) * qf_reward + replay_data.rewards

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer

        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

        critic_losses.append(critic_loss.item())

        # Compute regularization loss
        # sample action a ~ exp(Q(s, a))
        with th.no_grad():
            actions_expq, _ = self.actor.action_log_prob(replay_data.observations)
        regularization_loss = (th.min(th.cat(self.critic(replay_data.observations, actions_expq), dim=1),
                               dim=1, keepdim=True)[0] ** 2)

        regularization_loss = regularization_loss.mean()
        regularization_losses.append(regularization_loss.item())

        # Optimize the critic
        critic_loss += regularization_loss
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += 1
        return {"qf_loss": np.mean(critic_losses), "policy_loss": np.mean(actor_losses),
                "reg_loss": np.mean(regularization_losses), "ent_coef_loss": np.mean(ent_coef_losses),
                "ent_coef": ent_coef.item()}
