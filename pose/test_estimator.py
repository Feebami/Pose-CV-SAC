import omegaconf
import mani_skill.envs
import torch

from agent import PoseEstimator

def test_estimator():
    config = {
        'env_id': 'PickCube-v1',
        'resolution': 256,
        'channel_multiplier': 8,
        'max_channels': 128,
        'kernel_size': 3,
        'encoding_dim': 256,
        'pose': True,
        'camera_position': [0.3, 0, 0.3],
    }
    config = omegaconf.OmegaConf.create(config)
    estimator = PoseEstimator(config)

if __name__ == "__main__":
    test_estimator()