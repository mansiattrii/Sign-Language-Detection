"""Feature-group ablation: the same model trained on each feature configuration with several seeds.

    python -m src.ablation                 # LSTM -> results/ablation.csv, results/ablation.md
    python -m src.ablation --model gru     # -> results/ablation_gru.csv, results/ablation_gru.md
    python -m src.ablation --model rf

Each finished run is appended to the CSV immediately; rerunning skips runs already in it, so an
interrupted ablation resumes where it stopped. Use --fresh to start over.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import keras
import pandas as pd

from src.config import load_config
from src.evaluate import scores
from src.models import RECURRENT_CELLS, predict_proba
from src.train import fit, load_training_data

log = logging.getLogger("ablation")

METRICS = ["val_top1", "test_top1", "test_top5", "test_macro_f1"]


def run_one(kind: str, config_name: str, groups: list[str], seed: int, cfg: dict[str, Any],
            data: Any, split: dict[str, Any]) -> dict[str, Any]:
    bundle = fit(kind, data, split, cfg, groups, seed, f"{kind}_{config_name}_s{seed}")
    test = data.subset(split["test"])
    test_scores = scores(test.y, predict_proba(bundle, test.X))
    info = bundle["train_info"]
    row = {"model": kind, "config": config_name, "groups": "+".join(groups),
           "n_features": len(bundle["columns"]), "seed": seed,
           "epochs_run": info.get("epochs_run"), "best_epoch": info.get("best_epoch"),
           "val_top1": info["val_accuracy"], "test_top1": test_scores["top1_accuracy"],
           "test_top5": test_scores["top5_accuracy"], "test_macro_f1": test_scores["macro_f1"]}
    if kind in RECURRENT_CELLS:
        keras.backend.clear_session()
    return row


def summarize(df: pd.DataFrame, config_order: list[str]) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) of each metric per configuration."""
    grouped = df.groupby("config", sort=False)
    out = grouped[METRICS].agg(["mean", "std"])
    out.insert(0, ("n_features", ""), grouped["n_features"].first())
    out.insert(1, ("runs", ""), grouped.size())
    return out.reindex([c for c in config_order if c in out.index])


def to_markdown(df: pd.DataFrame, summary: pd.DataFrame, kind: str, n_test: int) -> str:
    lines = [f"# Feature ablation: {kind.upper()}", "",
             f"Test split: {n_test} INCLUDE clips from recording sittings not seen in training. "
             "Values are mean ± sample standard deviation over seeds.", "",
             "| Configuration | Features | Runs | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |",
             "|---|---|---|---|---|---|---|"]
    for config, row in summary.iterrows():
        cells = [f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}" for m in METRICS]
        lines.append(f"| {config} | {int(row[('n_features', '')])} | {int(row[('runs', '')])} | " + " | ".join(cells) + " |")
    lines += ["", "## Individual runs", "",
              "| Configuration | Seed | Epochs (best) | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |",
              "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        epochs = "" if pd.isna(r["epochs_run"]) else f"{int(r['epochs_run'])} ({int(r['best_epoch'])})"
        lines.append(f"| {r['config']} | {r['seed']} | {epochs} | {r['val_top1']:.3f} | {r['test_top1']:.3f} | "
                     f"{r['test_top5']:.3f} | {r['test_macro_f1']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["rf", *RECURRENT_CELLS], default="lstm")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default from config")
    parser.add_argument("--fresh", action="store_true", help="discard previous runs in the CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    seeds = args.seeds or cfg["ablation"]["seeds"]
    configs: dict[str, list[str]] = cfg["ablation"]["configs"]
    suffix = "" if args.model == "lstm" else f"_{args.model}"
    results = Path(cfg["paths"]["results_dir"])
    csv_path, md_path = results / f"ablation{suffix}.csv", results / f"ablation{suffix}.md"
    results.mkdir(parents=True, exist_ok=True)

    done = pd.read_csv(csv_path) if csv_path.exists() and not args.fresh else pd.DataFrame()
    finished = set(zip(done.get("config", []), done.get("seed", [])))
    data, split = load_training_data(cfg)

    rows = done.to_dict("records")
    for config_name, groups in configs.items():
        for seed in seeds:
            if (config_name, seed) in finished:
                log.info("skip %s seed %d (already in %s)", config_name, seed, csv_path)
                continue
            rows.append(run_one(args.model, config_name, groups, seed, cfg, data, split))
            pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
            log.info("%s seed %d: test top-1 %.3f", config_name, seed, rows[-1]["test_top1"])

    df = pd.DataFrame(rows)
    df = df[df["config"].isin(configs) & df["seed"].isin(seeds)]
    summary = summarize(df, list(configs))
    md_path.write_text(to_markdown(df, summary, args.model, len(split["test"])), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
