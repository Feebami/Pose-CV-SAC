import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Trainer class for Soft Actor-Critic (SAC) algorithm
class SACTrainer:
    # Initialize trainer with environment, config, and SAC networks
    def __init__(self, env, config, actor, q1, q2, q1_target, q2_target):
        self.config = config
        self.gamma = config.gamma  # Discount factor for future rewards
        self.tau = config.tau  # Soft update coefficient for target networks
        self.actor = actor  # Policy network
        self.q1 = q1  # First Q-network
        self.q2 = q2  # Second Q-network (for double Q-learning)
        self.q1_target = q1_target  # Target for first Q-network
        self.q2_target = q2_target  # Target for second Q-network
        # Optimizer for both Q-networks
        self.q_optimizer = AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.q_lr,
            weight_decay=config.weight_decay[0]
        )
        # Optimizer for actor (policy) network
        self.a_optimizer = AdamW(
            self.actor.parameters(),
            lr=config.a_lr,
            weight_decay=config.weight_decay[1]
        )
        # Learnable temperature parameter for entropy regularization
        self.log_alpha = torch.tensor([-1.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()  # Initial entropy coefficient
        # Optimizer for alpha
        self.alpha_optimizer = Adam([self.log_alpha], lr=config.q_lr)
        # Target entropy based on action space dimensionality
        self.target_entropy = -np.prod(env.single_action_space.shape)

    # Update Q-networks (critics) using Bellman backup
    def update_critics(self, data):
        # Compute target Q-values without gradients
        with torch.no_grad():
            # Sample next actions and log probs from policy
            next_state_actions, next_state_log_pi, _, _ = self.actor.get_action(data['next_obs'])
            # Get target Q-values from both target networks
            qf1_next_target = self.q1_target(data['next_obs'], next_state_actions)
            qf2_next_target = self.q2_target(data['next_obs'], next_state_actions)
            # Min of targets minus entropy term
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # Bellman equation: reward + discounted min target (if not done)
            next_q_value = data['rewards'].flatten() + (1 - data['dones'].flatten()) * self.gamma * (min_qf_next_target).view(-1)
        # Current Q-values for observed state-action pairs
        qf1_a_values = self.q1(data['obs'], data['actions']).view(-1)
        qf2_a_values = self.q2(data['obs'], data['actions']).view(-1)
        # MSE loss for each Q-network
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss  # Total critic loss

        # Compute absolute TD error for PER
        td_error = (qf1_a_values - next_q_value).abs().detach()

        # Optimize critics
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        # Gradient clipping to stabilize training
        clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # Return loss value and TD error
        return qf_loss.item(), td_error
    
    # Update actor (policy) to maximize expected Q-value minus entropy
    def update_actor(self, data):
        # Sample actions and log probs from current policy
        state_actions, log_pi, _, _ = self.actor.get_action(data['obs'])
        # Get Q-values for sampled actions
        qf1_a_values = self.q1(data['obs'], state_actions)
        qf2_a_values = self.q2(data['obs'], state_actions)
        # Min Q-value across both networks
        min_qf_a_values = torch.min(qf1_a_values, qf2_a_values)

        # Actor loss: entropy-regularized policy gradient
        actor_loss = (self.alpha * log_pi - min_qf_a_values).mean()

        # Optimize actor
        self.a_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping for stability
        clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.a_optimizer.step()

        # Return loss value
        return actor_loss.item()
    
    # Soft update target networks towards current Q-networks
    def update_target_networks(self):
        # Update Q1 target
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        # Update Q2 target
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    # Update entropy coefficient (alpha) to target entropy
    def update_alpha(self, data):
        # Compute log probs without gradients
        with torch.no_grad():
            _, log_pi, _, _ = self.actor.get_action(data['obs'])
        
        # Alpha loss: encourage policy entropy towards target
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Update alpha value
        self.alpha = self.log_alpha.exp().item()
