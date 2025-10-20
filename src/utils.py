import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def smape(y_true, y_pred, eps=1e-8):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return 100.0 * np.mean(numerator / denominator)

def mase(y_true, y_pred, seasonality=7):
    # naive seasonal forecast
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) <= seasonality:
        return np.nan
    naive = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    return np.mean(np.abs(y_true - y_pred)) / (naive + 1e-8)

def to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
