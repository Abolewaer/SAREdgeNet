from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch


def seed_training(config):
    seed = int(config.get('training_seed',42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segmentation_split(dataset,config):
    from .spatial_protocol import segmentation_indices
    return segmentation_indices(dataset, config)
