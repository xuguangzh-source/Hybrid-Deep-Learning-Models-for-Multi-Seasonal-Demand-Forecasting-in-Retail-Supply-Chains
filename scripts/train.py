import argparse, os
from hybrid_demand_forecasting.trainer import train_loop, load_configs
from hybrid_demand_forecasting.utils import ensure_dir, set_seed
import pandas as pd
import numpy as np

def make_synth_csv(path, n_stores=5, n_items=20, days=500, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    base_date = np.datetime64("2022-01-01")
    for s in range(n_stores):
        for i in range(n_items):
            level = rng.uniform(20, 200)
            trend = rng.uniform(-0.01, 0.05)
            season_week = rng.normal(0, 5, 7)
            for d in range(days):
                date = base_date + np.timedelta64(d, "D")
                # weekly seasonality + noise
                w = season_week[int(d % 7)]
                promo = 1 if rng.random() < 0.1 else 0
                holiday = 1 if rng.random() < 0.02 else 0
                price = float(np.clip(rng.normal(10, 1.5) - promo * 1.0, 5, 20))
                sales = float(np.clip(level + trend*d + w + promo*5 + holiday*8 + rng.normal(0, 3), 0, None))
                rows.append([str(date), s, i, sales, price, promo, holiday])
    df = pd.DataFrame(rows, columns=["date","store_id","item_id","sales","price","promo","holiday"])
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["configs/base_config.yaml","configs/model_config.yaml","configs/training_config.yaml"])
    parser.add_argument("--make_synth", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_configs(args.configs)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    if args.make_synth:
        make_synth_csv(cfg["data"]["raw_csv"])

    metrics = train_loop(args.configs)
    print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
