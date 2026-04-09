"""
Stage 2 — Exploratory Data Analysis
Selects the top correlated features, treats outliers, and saves the cleaned
dataset. Outputs visualisations to the configured directory.
"""
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

log = logging.getLogger(__name__)


# ── Feature selection ─────────────────────────────────────────────────────────

def select_top_features(df: pd.DataFrame, n: int) -> list:
    corr = df.corr()["Target"].drop("Target").abs()
    return corr.nlargest(n).index.tolist()


# ── Multicollinearity check ───────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = add_constant(df)
    vif = pd.DataFrame({
        "Feature": X.columns,
        "VIF":     [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif[vif["Feature"] != "const"].reset_index(drop=True)


# ── Outlier treatment ─────────────────────────────────────────────────────────

def replace_outliers(df: pd.DataFrame, method: str, threshold: float, strategy: str) -> pd.DataFrame:
    df_out       = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.difference(["Target"])

    for col in numeric_cols:
        if method == "iqr":
            q1, q3 = df_out[col].quantile([0.25, 0.75])
            iqr     = q3 - q1
            lo, hi  = q1 - threshold * iqr, q3 + threshold * iqr
        elif method == "zscore":
            mean, std = df_out[col].mean(), df_out[col].std()
            lo, hi    = mean - threshold * std, mean + threshold * std
        else:
            raise ValueError(f"Unknown outlier method: {method!r}")

        mask  = (df_out[col] < lo) | (df_out[col] > hi)
        n_out = int(mask.sum())
        if n_out == 0:
            continue

        if strategy == "median":
            df_out.loc[mask, col] = df_out[col].median()
        elif strategy == "mean":
            df_out.loc[mask, col] = df_out[col].mean()
        elif strategy == "iqr_bound":
            df_out.loc[df_out[col] < lo, col] = lo
            df_out.loc[df_out[col] > hi, col] = hi
        elif strategy == "remove":
            df_out = df_out[~mask]
        else:
            raise ValueError(f"Unknown outlier strategy: {strategy!r}")

        log.debug("  %-15s  replaced %d outliers", col, n_out)

    return df_out


# ── Visualisations ────────────────────────────────────────────────────────────

def save_plots(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=100)
    plt.close()

    # Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Target"], bins=30, kde=True)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=100)
    plt.close()

    # Top-4 feature–target scatter
    top4 = df.corr()["Target"].drop("Target").abs().nlargest(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flat, top4):
        ax.scatter(df[feat], df["Target"], alpha=0.3, s=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("Target")
    fig.suptitle("Top 4 Features vs Target")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_target_scatter.png"), dpi=100)
    plt.close()

    log.info("Plots saved → %s", output_dir)


# ── Main entry ────────────────────────────────────────────────────────────────

def run_eda(config: dict) -> pd.DataFrame:
    cfg_data = config["data"]
    cfg_eda  = config["eda"]

    df = pd.read_csv(cfg_data["raw_path"])
    log.info("Loaded raw dataset: %d rows × %d cols", *df.shape)
    log.info("Missing values: %d", int(df.isnull().sum().sum()))

    # Feature selection
    top_features = select_top_features(df, cfg_eda["top_n_features"])
    log.info("Top %d features selected: %s", len(top_features), top_features)
    df = df[top_features + ["Target"]].copy()

    # Multicollinearity
    vif = compute_vif(df.drop(columns=["Target"]))
    log.info("VIF results:\n%s", vif.to_string(index=False))

    # Outlier treatment
    df_clean = replace_outliers(
        df,
        method=cfg_eda["outlier_method"],
        threshold=cfg_eda["outlier_threshold"],
        strategy=cfg_eda["outlier_strategy"],
    )
    log.info("After outlier treatment: %d rows × %d cols", *df_clean.shape)

    # Visualisations
    save_plots(df_clean, cfg_eda["output_dir"])

    # Save cleaned dataset
    out = cfg_data["processed_path"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out, index=False)
    log.info("Saved cleaned dataset → %s", out)

    return df_clean


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    for search in [Path.cwd(), Path.cwd().parent]:
        cfg_path = search / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            break
    else:
        raise FileNotFoundError("config.yaml not found. Run from projects/ directory.")

    run_eda(cfg)
