import torch
import os
import numpy as np
import random

IMAGE_SIZT = 416
NUM_CLASS = 9
NUM_FEAT = 5
NUM_ATTR = NUM_CLASS + NUM_FEAT
ANCHOR = [
    (10, 33), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326) 
        ]
NUM_ANCHOR_PER_SCALE = len(ANCHOR) // 3




def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
