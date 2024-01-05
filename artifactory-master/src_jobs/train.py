# import sys
# sys.path.append("../artifactory/")

import warnings

import torch

# from artifact import Saw

# stop warnings
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Storing hyperparameters as a dictionary, because we can directly log this config dict to W&B.
CONFIG = dict(
    # width of window
    width=512,
    convolution_features=[256, 128, 64, 32],
    convolution_width=[5, 9, 17, 33],
    convolution_dropout=0.0,
    transformer_heads=2,
    transformer_feedforward=128,
    transformer_layers=2,
    transformer_dropout=0,
    loss="mask",
    loss_boost_fp=0,
    # artifact=Saw(min_width=4, max_width=32),
    # Optimizer Parameter
    # LearningRate Scheduler
    # parameters for study
    batch_size=32,  # 'values': [32, 64, 128]
    wandb_group_name="test_setup",
    wandb_project_name="artifactory",
)
