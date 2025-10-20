# 🧠 Hybrid Deep Learning Models for Multi-Seasonal Demand Forecasting in Retail Supply Chains

A production-ready repository implementing a **hybrid deep learning model** (TCN ➜ BiLSTM ➜ Attention) for **multi-seasonal demand forecasting** with exogenous variables (promotions, holidays, price, weather, CPI, etc.). Includes data pipeline, training loop, evaluation, explainability, and Optuna tuning.

## ✨ Highlights
- **Hybrid architecture**: Temporal Convolutional Networks (local patterns) + BiLSTM (long-range) + Attention (context weighting).
- **Multi-seasonality**: Fourier features + STL deseasonalization (optional) for weekly/monthly/annual cycles.
- **Exogenous variables**: Robust feature engineering for events, price elasticity, and weather/economic indicators.
- **Explainability**: Attention heatmaps + SHAP feature importance.
- **Reproducible**: YAML configs, deterministic seeds, logged artifacts and checkpoints.

---

## 📦 Repository Structure
```
Hybrid-Deep-Learning-Demand-Forecasting/
├── configs/
│   ├── base_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── results/
│   ├── checkpoints/
│   ├── figures/
│   └── logs/
├── scripts/
│   └── train.py
├── src/
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── feature_engineer.py
│   ├── trainer.py
│   ├── utils.py
│   └── models/
│       ├── attention_layer.py
│       ├── hybrid_model.py
│       └── tcn_layer.py
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🚀 Quickstart

```bash
# 1) Create env and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Organize your CSV under data/raw/sales.csv with columns:
#    ['date','store_id','item_id','sales','price','promo','holiday']
#    (you can add extra columns; they will be encoded)
#    Example synthetic dataset generator is provided (scripts/train.py --make_synth 1)
#
# 3) Train with default configs
python scripts/train.py --configs configs/base_config.yaml configs/model_config.yaml configs/training_config.yaml

# 4) Evaluate and plot figures
# (Evaluation runs automatically after training; figures saved under results/figures/)
```

### Example: make a synthetic dataset and train
```bash
python scripts/train.py --make_synth 1 --epochs 5
```

---

## 🧠 Method (TCN ➜ BiLSTM ➜ Attention)
1. **TCN** learns local temporal patterns with dilated causal convolutions.
2. **BiLSTM** captures long-range dependencies bidirectionally.
3. **Attention** weights influential time steps before dense forecast head.

> Loss = Smooth L1 + seasonal regularization (encourages weekly/annual smoothness).

---

## 📊 Metrics & Outputs
- `MAE`, `RMSE`, `sMAPE`, `MASE` (per series + aggregated).
- Plots: forecast vs actual, residual seasonality, attention heatmaps, SHAP importances.
- Artifacts: best checkpoint (`results/checkpoints/best.ckpt`) and TensorBoard logs in `results/logs`.

---

## ⚙️ Configs
- `base_config.yaml`: data paths, column names, sequence/forecast lengths.
- `model_config.yaml`: channels, LSTM hidden size, attention size, dropout.
- `training_config.yaml`: optimizer, LR scheduler, early stopping, Optuna settings.

---

## 🔬 Optuna Tuning
```bash
python scripts/train.py --tune 1 --n_trials 20
```

---

## 📜 Citation
```
@article{zhang2025hybridforecast,
  title={Hybrid Deep Learning Models for Multi-Seasonal Demand Forecasting in Retail Supply Chains},
  author={Zhang, Xuguang},
  year={2025}
}
```

MIT © 2025 Xuguang Zhang
