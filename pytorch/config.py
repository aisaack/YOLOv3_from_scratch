import torch
import os
import numpy as np
import random

class Config:
    DEVICE = 'cuda' if torch.cuda.is_availabel() else 'cpu'
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    INPUT_CHANNEL = 3
    INPUT_IMAGE = 416    # it could be 608
    S = [INPUT_IMAGE // 32, INPUT_IMAGE // 16, INPUT_IMAGE // 8]
    CONFIDENCE_THRESHOLD = .05

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
