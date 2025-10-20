import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx], "id": self.ids[idx]}

def build_dataloaders(cfg):
    csv_path = cfg["data"]["raw_csv"]
    df = pd.read_csv(csv_path, parse_dates=[cfg["data"]["date_col"]])
    date_col = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]
    id_cols = cfg["data"]["id_cols"]
    exo_cols = cfg["data"]["exogenous_cols"]
    seq_len = cfg["data"]["seq_len"]
    horizon = cfg["data"]["forecast_horizon"]

    # sort and fill
    df = df.sort_values(by=id_cols + [date_col]).reset_index(drop=True)
    df[target_col] = df[target_col].astype(float).fillna(0.0)
    for c in exo_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype(float).fillna(0.0)

    # scaling per series (store_id, item_id)
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()

    X_list, y_list, id_list = [], [], []
    for (sid, iid), g in df.groupby(id_cols):
        g = g.sort_values(date_col)
        y = g[target_col].values.reshape(-1, 1)
        X_exo = g[exo_cols].values

        y_scaled = scaler_y.fit_transform(y).flatten()
        X_scaled = scaler_x.fit_transform(np.hstack([y, X_exo]))  # include lagged y implicitly

        # build windows
        for t in range(len(g) - seq_len - horizon + 1):
            X_list.append(X_scaled[t : t + seq_len, :])
            y_list.append(y_scaled[t + seq_len : t + seq_len + horizon])
            id_list.append((sid, iid))

    X = np.stack(X_list) if X_list else np.zeros((0, seq_len, len(exo_cols)+1))
    y = np.stack(y_list) if y_list else np.zeros((0, horizon))

    n = X.shape[0]
    n_train = int(n * cfg["data"]["train_ratio"])
    n_valid = int(n * cfg["data"]["valid_ratio"])
    idx_train = slice(0, n_train)
    idx_valid = slice(n_train, n_train + n_valid)
    idx_test = slice(n_train + n_valid, n)

    train_ds = TimeSeriesDataset(X[idx_train], y[idx_train], id_list[:n_train])
    valid_ds = TimeSeriesDataset(X[idx_valid], y[idx_valid], id_list[n_train:n_train+n_valid])
    test_ds  = TimeSeriesDataset(X[idx_test],  y[idx_test],  id_list[n_train+n_valid:])

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, num_workers=cfg["train"]["num_workers"])

    return make_loader(train_ds, True), make_loader(valid_ds, False), make_loader(test_ds, False), X.shape[-1]
