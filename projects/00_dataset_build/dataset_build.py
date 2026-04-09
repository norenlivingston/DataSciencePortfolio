"""
Stage 1 — Dataset Build
Generates a synthetic regression dataset and writes it to disk.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_dataset(config: dict) -> pd.DataFrame:
    cfg        = config["data"]
    n_samples  = cfg["n_samples"]
    n_features = cfg["n_features"]
    seed       = cfg["random_seed"]
    out_path   = cfg["raw_path"]

    log.info("Generating dataset  (n_samples=%d, n_features=%d, seed=%d)",
             n_samples, n_features, seed)

    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_samples, n_features))

    # Add high noise to 3 randomly chosen features to simulate real-world messiness
    noisy_idx   = rng.choice(n_features, 3, replace=False)
    X[:, noisy_idx] += rng.standard_normal((n_samples, 3)) * 5

    # 8 features are strongly correlated with the target
    corr_idx     = rng.choice(n_features, 8, replace=False)
    coefficients = rng.uniform(40, 90, 8)
    X[:, corr_idx] += rng.standard_normal((n_samples, 8)) * 0.3

    y = X[:, corr_idx] @ coefficients + rng.standard_normal(n_samples) * 1.5

    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
    df["Target"] = y

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved raw dataset → %s  (%d rows × %d cols)", out_path, *df.shape)

    return df


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Support running from either projects/ or projects/00_dataset_build/
    for search in [Path.cwd(), Path.cwd().parent]:
        cfg_path = search / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            break
    else:
        raise FileNotFoundError("config.yaml not found. Run from projects/ directory.")

    build_dataset(cfg)
