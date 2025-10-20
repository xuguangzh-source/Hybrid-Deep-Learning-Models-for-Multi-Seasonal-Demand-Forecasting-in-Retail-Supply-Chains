import os, time, yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .utils import set_seed, ensure_dir, to_device
from .data_loader import build_dataloaders
from .models.hybrid_model import HybridTCNBiLSTMAttention
from .evaluator import evaluate

def load_configs(config_paths):
    cfg = {}
    for p in config_paths:
        with open(p, "r") as f:
            part = yaml.safe_load(f)
            # shallow merge
            for k,v in part.items():
                if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    return cfg

def make_model(cfg, input_dim):
    mcfg = cfg["model"]
    return HybridTCNBiLSTMAttention(
        input_dim=input_dim,
        tcn_channels=mcfg["tcn_channels"],
        tcn_kernel_size=mcfg["tcn_kernel_size"],
        tcn_dropout=mcfg["tcn_dropout"],
        lstm_hidden=mcfg["lstm_hidden"],
        lstm_layers=mcfg["lstm_layers"],
        attention_dim=mcfg["attention_dim"],
        dense_hidden=mcfg["dense_hidden"],
        horizon=cfg["data"]["forecast_horizon"],
        dropout=mcfg["dropout"],
    )

def train_loop(config_paths):
    cfg = load_configs(config_paths)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, input_dim = build_dataloaders(cfg)
    model = make_model(cfg, input_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
    loss_fn = nn.SmoothL1Loss()

    ensure_dir("results/checkpoints")
    ensure_dir("results/logs")
    ensure_dir("results/figures")
    writer = SummaryWriter(log_dir="results/logs") if cfg["train"].get("tensorboard", True) else None

    best_val = float("inf")
    patience = cfg["train"]["early_stopping_patience"]
    wait = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            opt.zero_grad()
            y_pred, attn = model(batch["x"])
            loss = loss_fn(y_pred, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            opt.step()
            total_loss += loss.item()
        scheduler.step(epoch + total_loss)

        # validation
        metrics = evaluate(model, valid_loader, device)
        val_loss = metrics["RMSE"]
        if writer:
            writer.add_scalar("train/loss", total_loss / max(1, len(train_loader)), epoch)
            writer.add_scalar("valid/RMSE", val_loss, epoch)
            writer.add_scalar("valid/MAE", metrics["MAE"], epoch)
            writer.add_scalar("valid/sMAPE", metrics["sMAPE"], epoch)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), "results/checkpoints/best.ckpt")
        else:
            wait += 1
            if wait >= patience:
                break

    # test evaluation
    model.load_state_dict(torch.load("results/checkpoints/best.ckpt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    if writer:
        for k, v in test_metrics.items():
            writer.add_scalar(f"test/{k}", v)
        writer.close()

    # Save metrics
    with open("results/metrics.json", "w") as f:
        import json
        json.dump(test_metrics, f, indent=2)

    return test_metrics
