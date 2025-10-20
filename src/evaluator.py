import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .utils import smape, mase, to_device

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    for batch in loader:
        batch = to_device(batch, device)
        y_pred, _ = model(batch["x"])
        y_true_all.append(batch["y"].cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
    sm = smape(y_true.flatten(), y_pred.flatten())
    ms = mase(y_true.flatten(), y_pred.flatten(), seasonality=7)
    return {"MAE": float(mae), "RMSE": float(rmse), "sMAPE": float(sm), "MASE": float(ms)}
