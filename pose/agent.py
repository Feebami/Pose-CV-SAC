import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import sapien_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Optimize for high precision matrix multiplications
torch.set_float32_matmul_precision('high')

def pose_loss(pred, target):
    pred_pos, pred_q = pred[..., :3], pred[..., 3:]
    tgt_pos, tgt_q = target[..., :3], target[..., 3:]

    # Normalize quaternions
    pred_q = F.normalize(pred_q, dim=-1)
    tgt_q = F.normalize(tgt_q, dim=-1)

    # Position MSE
    pos_loss = F.mse_loss(pred_pos, tgt_pos)

    # Quaternion geodesic-ish loss: min(||q - t||, ||q + t||)
    q_diff_1 = F.mse_loss(pred_q, tgt_q, reduction='none').sum(-1)
    q_diff_2 = F.mse_loss(pred_q, -tgt_q, reduction='none').sum(-1)
    rot_loss = torch.mean(torch.minimum(q_diff_1, q_diff_2))

    return pos_loss, rot_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): # of epochs to wait after last improvement.
            min_delta (float): Minimum improvement in monitored metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")   # for loss; use -inf for accuracy
        self.wait = 0
        self.stopped_epoch = None

    def step(self, current):
        # current is the latest val_loss (or metric)
        improved = current < (self.best - self.min_delta)
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True  # signal to stop
        return False


# Single convolution block
class SingleConv(nn.Sequential):
    # Initialize with input/output channels and kernel size
    def __init__(self, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=padding,
                bias=False
            ), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        super().__init__(*layers)
    
# Estimator for processing RGB images into pose estimates
class PoseEstimator(nn.Module):
    # Initialize with config for channels, conv type, etc.
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        _channels = [4, 8, 16, 32]  # Base channel progression
        channels = [min(int(c * config.channel_multiplier), config.max_channels) 
                    for c in _channels]
        output_channels = 7 if config.pose else 3  # Full pose or just position
        
        self.transform = v2.Compose([
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # HWC to CHW
            v2.ToDtype(torch.float32, scale=True)  # Scale to [0, 1]
        ])
        
        conv = SingleConv
        # Input convolutional block followed by pooling
        self.input_block = nn.Sequential(
            conv(3, channels[0], kernel_size=5),
            nn.MaxPool2d(2)
        )

        # Backbone: series of conv blocks and poolings
        self.backbone = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.backbone.append(conv(
                channels[i], 
                channels[i+1], 
                kernel_size=config.kernel_size
            ))
            self.backbone.append(nn.MaxPool2d(2))

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Full estimator sequence
        self.estimator = nn.Sequential(
            # Network
            self.input_block,
            *self.backbone,
            nn.Conv2d(channels[-1], channels[-1], kernel_size=1), nn.ReLU(True),
            self.pool,
            nn.Flatten(),
            nn.Linear(channels[-1], config.encoding_dim), nn.ReLU(True),
            nn.LayerNorm(config.encoding_dim),
            nn.Linear(config.encoding_dim, output_channels, bias=False)
        )
        self._train_estimator()

    def _train_estimator(self):
        print("Training Pose Estimator...")
        self.to(device)
        self.train()
        env = gym.make(
            self.config.env_id, 
            num_envs=self.config.estimator_batch_size, 
            obs_mode='state_dict+rgb', 
            control_mode='pd_ee_delta_pose', 
            reward_mode='normalized_dense',
            sensor_configs={
                'width': self.config.resolution, 
                'height': self.config.resolution,
                'pose': sapien_utils.look_at(eye=self.config.camera_position, target=[-0.1, 0, 0.1])
            }
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        early_stopping = EarlyStopping(patience=100, min_delta=1e-4)
        loss = float('inf')
        while not early_stopping.step(loss):
            obs, _ = env.reset()
            rgb = obs['sensor_data']['base_camera']['rgb']
            target = obs['extra']['obj_pose'] if self.config.pose else obs['extra']['obj_pose'][..., :3]
            pred = self(rgb)
            if self.config.pose:
                pos_loss, rot_loss = pose_loss(pred, target)
                loss = pos_loss + 0.1 * rot_loss
                print(f"Position Loss: {pos_loss.item():.6f}, Rotation Loss: {rot_loss.item():.6f}", end='\r')
            else:
                loss = F.mse_loss(pred, target)
                print(f"Position Loss: {loss.item():.6f}", end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        env.close()
        self.eval()
        # Freeze estimator parameters after training
        for param in self.parameters():
            param.requires_grad = False
        print(f"\nFinal Estimator Loss: {loss.item():.6f}")

    # Forward pass: transform and encode RGB input
    def forward(self, rgb):
        rgb = self.transform(rgb)
        return self.estimator(rgb)
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hidden_sizes[0] == hidden_sizes[-1], 'Unmatched risidual connection sizes'

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0], bias=False), 
            nn.LayerNorm(hidden_sizes[0]), 
            nn.ReLU(True)
        )

        self.blocks = nn.ModuleList()
        for j in range(num_blocks):
            block = nn.Sequential()
            for i in range(len(hidden_sizes) - 1):
                block.add_module(f'linear_{i}', nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False))
                block.add_module(f'layernorm_{i}', nn.LayerNorm(hidden_sizes[i+1]))
                block.add_module(f'relu_{i}', nn.ReLU(True))
            self.blocks.append(block)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = x + block(x)
        return x

# Q-network for Soft Actor-Critic with visual encoding
class SoftQ(nn.Module):
    # Initialize with config for encoder and MLP
    def __init__(self, env, state_dim, config):
        super().__init__()

        pose_size = 7 if config.pose else 3
        self.mlp = MLP(
            pose_size + state_dim + np.prod(env.single_action_space.shape),
            config.mlp_layers,
            config.mlp_blocks
        )
        self.output = nn.Linear(config.mlp_layers[-1], 1)

    # Forward pass: encode RGB, concat with action/state, compute Q
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = self.mlp(x)
        return self.output(x)
    
# Constants for clamping log standard deviation
LOG_STD_MAX = 2
LOG_STD_MIN = -10

# Actor (policy) network for SAC: Gaussian policy with visual encoding
class Actor(nn.Module):
    # Initialize with config for encoder and MLP
    def __init__(self, env, state_dim, config):
        super().__init__()

        pose_size = 7 if config.pose else 3        
        # MLP for combined encoding + state
        self.mlp = MLP(
            pose_size + state_dim, 
            config.mlp_layers,
            config.mlp_blocks
        )

        # Linear layers for mean and log_std of action distribution
        self.fc_mean = nn.Linear(config.mlp_layers[-1], np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(config.mlp_layers[-1], np.prod(env.single_action_space.shape))
        # Action space bounds for scaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    # Forward pass: compute mean and clamped log_std
    def forward(self, obs):
        x = obs
        x = self.mlp(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std
    
    # Get deterministic (mean) actions for evaluation (no gradients)
    @torch.no_grad()
    def get_eval_action(self, obs):
        x = obs
        x = self.mlp(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    # Sample stochastic actions with reparameterization and log probs
    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()  # Standard deviation
        # Normal distribution for sampling
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        y_t = torch.tanh(x_t)  # Squash to [-1, 1]
        # Scale and bias to action space
        action = y_t * self.action_scale + self.action_bias
        # Log probability with Jacobian correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # Scaled mean for reference
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std