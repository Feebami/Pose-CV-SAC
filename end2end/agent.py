# Core PyTorch modules for neural networks
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Optimize for high precision matrix multiplications
torch.set_float32_matmul_precision('high')

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
                padding=padding
            ), 
            nn.ReLU(True)
        ]
        super().__init__(*layers)
    
# Encoder for processing RGB images into latent features
class Encoder(nn.Module):
    # Initialize with config for channels, conv type, etc.
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _channels = [4, 8, 16, 32]  # Base channel progression
        channels = [min(int(c * config.channel_multiplier), config.max_channels) 
                    for c in _channels]
        
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

        # Full encoder sequence
        self.encoder = nn.Sequential(
            # Network
            self.input_block,
            *self.backbone,
            nn.Conv2d(channels[-1], channels[-1], kernel_size=1), nn.ReLU(True),
            self.pool,
            nn.Flatten(),
            nn.Linear(channels[-1], config.encoding_dim)
        )

    # Forward pass: transform and encode RGB input
    def forward(self, rgb):
        rgb = self.transform(rgb)
        return self.encoder(rgb)
    
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

        # Encoder for RGB, MLP for combined features
        self.encoder = Encoder(config)
        self.mlp = MLP(
            config.encoding_dim + state_dim + np.prod(env.single_action_space.shape),
            config.mlp_layers,
            config.mlp_blocks
        )
        self.output = nn.Linear(config.mlp_layers[-1], 1)

    # Forward pass: encode RGB, concat with action/state, compute Q
    def forward(self, obs, action):
        rgb = obs['rgb']
        state = obs['state']
        visual_encoding = self.encoder(rgb)
        x = torch.cat([visual_encoding, action, state], dim=1)
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

        # Encoder for RGB
        self.encoder = Encoder(config)
        
        # MLP for combined encoding + state
        self.mlp = MLP(
            config.encoding_dim + state_dim, 
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
        rgb = obs['rgb']
        state = obs['state']
        visual_encoding = self.encoder(rgb)
        x = torch.cat([visual_encoding, state], dim=1)
        x = self.mlp(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std
    
    # Get deterministic (mean) actions for evaluation (no gradients)
    @torch.no_grad()
    def get_eval_action(self, obs):
        rgb = obs['rgb']
        state = obs['state']
        x = self.encoder(rgb)
        x = torch.cat([x, state], dim=1)
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