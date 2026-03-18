import os
import random
import shutil
from time import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import ManiSkill and related wrappers for RGB-based environments
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils, sapien_utils
from mani_skill.utils.wrappers import RecordEpisode, \
    FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Configuration and tensor utilities
import argparse
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from tensordict import TensorDict
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from torchvision.transforms import v2
from tqdm import trange

# Custom SAC components for RGB inputs
from .agent import Actor, SoftQ, PoseEstimator
from common.trainer import SACTrainer

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Optimize for high precision matrix multiplications
torch.set_float32_matmul_precision('high')

# Main execution block
if __name__ == '__main__':
    start_time_global = time()
    # Parse command-line arguments for config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the config YAML file.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()

    # Set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load configuration from YAML file
    config = OmegaConf.load(args.config)

    # Extract key config parameters
    num_envs = config.num_envs
    buffer_size = config.buffer_size
    env_id = config.env_id
    global_steps = config.global_steps
    batch_size = config.batch_size
    utd = config.utd
    resolution = config.resolution
    camera_position = config.camera_position
    buffer_device = config.buffer_device
    buffer_alpha = config.buffer_alpha

    # Calculate gradient steps per iteration
    grad_steps_per_iter = int(num_envs * utd)
    # Sensor configs for RGB resolution
    sensor_configs = {
        'width': resolution, 
        'height': resolution, 
        'pose': sapien_utils.look_at(eye=camera_position, target=[-0.1, 0, 0.1])
    }

    pose_estimator = PoseEstimator(config).to(device)
    pose_estimator = torch.compile(pose_estimator, mode='default')
    print(f'Pose estimator parameters: {sum(p.numel() for p in pose_estimator.parameters()):,}')

    # Create vectorized training environment with RGB obs
    env = gym.make(
        env_id, 
        num_envs=num_envs, 
        obs_mode='rgb', 
        control_mode='pd_ee_delta_pose', 
        reward_mode='normalized_dense',
        sensor_configs=sensor_configs,
    )

    env = FlattenRGBDObservationWrapper(env)
    env = ManiSkillVectorEnv(
        env, 
        num_envs=num_envs, 
        ignore_terminations=True, 
        auto_reset=False
    )
    # Print simulation details for debugging
    env.unwrapped.print_sim_details()
    # Determine steps per episode and total episodes
    steps_per_episode = gym_utils.find_max_episode_steps_value(env)
    episodes = global_steps // (num_envs * steps_per_episode)
    buffer_init_episodes = batch_size // (steps_per_episode * num_envs) + 1

    # Initialize prioritized replay buffer
    buffer = TensorDictPrioritizedReplayBuffer(
        alpha=buffer_alpha,
        beta=0.4,
        storage=LazyTensorStorage(buffer_size, device=buffer_device), 
        batch_size=batch_size, 
        pin_memory=True, 
        prefetch=4,
        priority_key='key'
    )

    # Fill buffer with initial random actions for exploration
    start_fill = time()
    for _ in range(buffer_init_episodes):
        obs, info = env.reset()
        with torch.no_grad():
            pose_hat = pose_estimator(obs['rgb'])
        obs = torch.cat([pose_hat, obs['state']], dim=-1)

        for _ in range(steps_per_episode):
            actions = env.action_space.sample()  # Sample random actions
            next_obs, rewards, dones, truncated, info = env.step(actions)
            next_obs = torch.cat([pose_hat, next_obs['state']], dim=-1)
            # Create TensorDict for experience replay
            data = TensorDict({
                "obs": obs,
                "actions": actions,
                "next_obs": next_obs,
                "rewards": rewards.unsqueeze(1),
                "dones": dones.unsqueeze(1).float(),
                'key': torch.ones((num_envs, 1))  # Initial priority keys
            }, batch_size=num_envs)

            # Add data to buffer
            buffer.extend(data)

            obs = next_obs
    fill_time = time() - start_fill
    # Log buffer filling performance
    print(
        f'{len(buffer)} steps in {fill_time} seconds: '
        f'{(len(buffer)/fill_time):.2f} steps/second'
    )

    # Generate unique run name based on env and timestamp
    run_name = f'control-{env_id}-{int(time())}-{args.seed}'
    # Set up directory for video outputs
    render_output_dir = f'./runs/{run_name}/videos'
    writer = SummaryWriter(f'runs/{run_name}')
    # Copy the config file into the run directory for reproducibility
    os.makedirs(f'runs/{run_name}', exist_ok=True)
    shutil.copy(args.config, f'runs/{run_name}/config.yaml')
    writer.add_text('config', OmegaConf.to_yaml(config))

    # Create evaluation environment
    eval_env = gym.make(
        env_id, 
        num_envs=batch_size, 
        obs_mode='rgb', 
        control_mode='pd_ee_delta_pose', 
        reward_mode='normalized_dense',
        sensor_configs=sensor_configs
    )

    eval_env = FlattenRGBDObservationWrapper(eval_env)
    eval_env = ManiSkillVectorEnv(
        eval_env, 
        num_envs=batch_size, 
        ignore_terminations=True, 
        auto_reset=False
    )
    # Create rendering environment
    render_env = gym.make(
        env_id, 
        num_envs=16,
        obs_mode='rgb', 
        control_mode='pd_ee_delta_pose', 
        reward_mode='sparse',
        sensor_configs=sensor_configs,
        render_mode='rgb_array'
    )
    render_env = FlattenRGBDObservationWrapper(render_env)
    render_env = ManiSkillVectorEnv(
        render_env, 
        num_envs=16, 
        ignore_terminations=True, 
        auto_reset=False
    )
    # Wrap env to record episodes as videos
    render_env = RecordEpisode(
        render_env,
        output_dir=render_output_dir,
        avoid_overwriting_video=True,
        save_trajectory=False,
        max_steps_per_video=steps_per_episode,
        video_fps=25
    )

    sample_obs, _ = env.reset()
    state_dim = sample_obs['state'].shape[1]

    # Initialize actor and Q-networks
    actor = Actor(env, state_dim, config).to(device)
    q1 = SoftQ(env, state_dim, config).to(device)
    q2 = SoftQ(env, state_dim, config).to(device)
    q1_target = SoftQ(env, state_dim, config).to(device)
    q2_target = SoftQ(env, state_dim, config).to(device)
    # Sync target networks with initial Q-network states
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # Compile models for optimized performance
    if device == 'cuda':
        actor = torch.compile(actor, mode='default')
        q1 = torch.compile(q1, mode='default')
        q2 = torch.compile(q2, mode='default')
        q1_target = torch.compile(q1_target, mode='default')
        q2_target = torch.compile(q2_target, mode='default')
        
    # Initialize SAC trainer
    trainer = SACTrainer(
        env=env,
        config=config,
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
    )

    # Count and print model parameters for reference
    print(
        f'Actor parameters: {sum(p.numel() for p in actor.parameters()):,}, '
        f'Critic parameters: {sum(p.numel() for p in q1.parameters()):,}'
    )

    # Training loop setup
    total_steps = 0
    result_list = []
    train_episode_times = []
    pbar = trange(episodes)
    pbar.set_description(run_name)
    for episode in pbar:
        # Set models to training mode
        actor.train()
        q1.train()
        q2.train()
        obs, _ = env.reset()
        with torch.no_grad():
            pose_hat = pose_estimator(obs['rgb'])
        obs = torch.cat([pose_hat, obs['state']], dim=-1)

        # Collect data for one episode
        episode_start_time = time()
        for step in range(steps_per_episode):
            with torch.no_grad():
                # Get actions from actor (with exploration)
                actions, log_probs, means, stds = actor.get_action(obs)
                actions = actions.detach()
            # Step environment
            next_obs, rewards, dones, truncated, info = env.step(actions)
            next_obs = torch.cat([pose_hat, next_obs['state']], dim=-1)
            total_steps += num_envs
            # Create TensorDict for new experiences
            data = TensorDict({
                "obs": obs,
                "actions": actions,
                "next_obs": next_obs,
                "rewards": rewards.unsqueeze(1),
                "dones": dones.unsqueeze(1).float(),
                'key': torch.ones((num_envs, 1))  # Priority keys
            }, batch_size=num_envs)

            # Add to buffer
            buffer.extend(data)

            obs = next_obs

            # Perform gradient updates
            for _ in range(grad_steps_per_iter):
                # Sample batch from buffer
                sample = buffer.sample().to(device, non_blocking=True)

                # Update critics, actor, targets, and alpha
                qf_loss, td_error = trainer.update_critics(sample)
                actor_loss = trainer.update_actor(sample)
                trainer.update_target_networks()
                trainer.update_alpha(sample)

                # Update sample priorities based on TD error
                sample.set('key', td_error)
                buffer.update_tensordict_priority(sample)

        # Log histograms and scalars
        writer.add_histogram('actions', actions, total_steps)
        writer.add_histogram('log_probs', log_probs, total_steps)
        writer.add_histogram('means', means, total_steps)
        writer.add_histogram('stds', stds, total_steps)
        writer.add_histogram('rewards', rewards, total_steps)
        writer.add_scalar('qf_loss', qf_loss, total_steps)
        writer.add_scalar('actor_loss', actor_loss, total_steps)
        writer.add_scalar('alpha', trainer.alpha, total_steps)

        episode_time = time() - episode_start_time
        train_episode_times.append(episode_time)
        avg_episode_time = np.mean(train_episode_times)

        # Evaluate policy every 10 episodes
        if not (episode + 1) % 10:
            actor.eval()  # Set actor to evaluation mode
            eval_rewards = torch.zeros((steps_per_episode, batch_size))
            eval_successes = torch.zeros((steps_per_episode, batch_size))
            action_times = []
            eval_obs, _ = eval_env.reset()
            with torch.no_grad():
                pose_hat = pose_estimator(eval_obs['rgb'])
            eval_obs = torch.cat([pose_hat, eval_obs['state']], dim=-1)
            for step in range(steps_per_episode):
                # Get deterministic actions for evaluation
                action_decision_start = time()
                actions = actor.get_eval_action(eval_obs)
                action_decision_time = time() - action_decision_start
                action_times.append(action_decision_time)
                eval_obs, rewards, _, _, infos = eval_env.step(actions)
                eval_obs = torch.cat([pose_hat, eval_obs['state']], dim=-1)
                eval_rewards[step, :] = rewards.squeeze().cpu()
                eval_successes[step, :] = torch.tensor(
                    infos['success'],
                    dtype=torch.float32
                )
            # Compute average returns and success rates
            avg_return = eval_rewards.sum(dim=0).mean().item()
            episode_successes = eval_successes[-1]
            success_rate = episode_successes.mean().item()
            anytime_successes = eval_successes.any(dim=0)
            anytime_success_rate = anytime_successes.float().mean().item()
            writer.add_scalar('eval/avg_return', avg_return, total_steps)
            writer.add_scalar('eval/success_rate', success_rate, total_steps)
            writer.add_scalar(
                'eval/anytime_success_rate', 
                anytime_success_rate, 
                total_steps
            )
            avg_action_time = np.mean(action_times) * 1000  # in ms
            writer.add_scalar(
                'eval/avg_action_time_ms', 
                avg_action_time, 
                total_steps
            )

            current_wall_time = time() - start_time_global
            max_memory_GB = torch.cuda.max_memory_allocated() / (1024 ** 3) \
                if torch.cuda.is_available() else 0.0
            result_list.append({
                'step': total_steps,
                'avg_return': avg_return,
                'success_rate': success_rate,
                'anytime_success_rate': anytime_success_rate,
                'avg_action_time_ms': avg_action_time,
                'avg_episode_time_s': avg_episode_time,
                'wall_time_s': current_wall_time,
                'max_memory_GB': max_memory_GB,
            })

            # Render episodes every 40 episodes
            if not (episode + 1) % 40:
                eval_obs, _ = render_env.reset()
                with torch.no_grad():
                    pose_hat = pose_estimator(eval_obs['rgb'])
                eval_obs = torch.cat([pose_hat, eval_obs['state']], dim=-1)
                for _ in range(steps_per_episode):
                    actions = actor.get_eval_action(eval_obs)
                    eval_obs, rewards, dones, truncated, info = \
                        render_env.step(actions)
                    eval_obs = torch.cat([pose_hat, eval_obs['state']], dim=-1)

    # Clean up environments and writer
    results_df = pd.DataFrame(result_list)
    results_df.to_csv(f'runs/{run_name}/results.csv', index=False)
    env.close()
    eval_env.close()
    render_env.close()
    writer.close()