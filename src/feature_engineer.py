import pandas as pd
import numpy as np

def add_fourier_terms(df: pd.DataFrame, date_col: str, periods=(7, 30, 365), K=3):
    df = df.copy()
    t = pd.to_datetime(df[date_col])
    for p in periods:
        for k in range(1, K+1):
            df[f"fourier_sin_{p}_{k}"] = np.sin(2*np.pi*k*t.dt.dayofyear / p)
            df[f"fourier_cos_{p}_{k}"] = np.cos(2*np.pi*k*t.dt.dayofyear / p)
    return df

def preprocess(df: pd.DataFrame, date_col: str):
    df = df.copy()
    # Fill missing
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna(method="ffill").fillna(method="bfill")
    return df
