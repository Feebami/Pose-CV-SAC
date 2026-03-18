import omegaconf
import mani_skill.envs
import torch

from .agent import PoseEstimator

def test_estimator():
    config = {
        'env_id': 'PickCube-v1',
        'resolution': 128,
        'channel_multiplier': 8,
        'max_channels': 128,
        'kernel_size': 3,
        'encoding_dim': 256,
        'pose': False,
        'camera_position': [0.3, 0, 0.3],
        'estimator_batch_size': 256,
    }
    config = omegaconf.OmegaConf.create(config)
    estimator = PoseEstimator(config)

if __name__ == "__main__":
    test_estimator()