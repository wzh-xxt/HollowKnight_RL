import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_tensor(x):
    x = np.array(x)

    # (B, T, H, W, C) → (B, C, T, H, W)
    if len(x.shape) == 5:
        x = x.transpose(0, 4, 1, 2, 3)
        x = x / 255.0
    elif len(x.shape) == 4:
        x = x.transpose(3, 0, 1, 2)  # 单样本
        x = x[None, ...]
        x = x / 255.0
    return torch.tensor(x, dtype=torch.float32).to(device)