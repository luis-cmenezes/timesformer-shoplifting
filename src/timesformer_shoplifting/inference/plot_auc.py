"""
Plota eval/auc_roc de todos os experimentos TimeSformer lado a lado.

Organizado em grid: linhas = num_frames (8, 32, 64), colunas = checkpoint (k400, ssv2).
Cada subplot compara unfreeze_head vs unfreeze_all.

Uso:
  uv run plot_auc.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results/timesformer")
METRIC = "eval/auc_roc"

# Organizar experimentos por (checkpoint, frames)
experiments = {}
for exp_dir in sorted(RESULTS_DIR.iterdir()):
    csv = exp_dir / "final_model" / "tb_metrics.csv"
    if not csv.exists():
        continue
    name = exp_dir.name  # e.g. timesformer-base-finetuned-k400_frames8_unfreeze_head
    parts = name.split("_")
    # checkpoint: k400 ou ssv2
    checkpoint = "k400" if "k400" in name else "ssv2"
    # frames
    frames = int([p for p in parts if p.startswith("frames")][0].replace("frames", ""))
    # strategy
    strategy = "head" if "unfreeze_head" in name else "all"

    df = pd.read_csv(csv)
    df = df[df[METRIC].notna()].copy()
    df["epoch"] = range(1, len(df) + 1)

    key = (checkpoint, frames)
    if key not in experiments:
        experiments[key] = {}
    experiments[key][strategy] = df

# Grid: linhas = frames, colunas = checkpoint
frame_sizes = [8, 32, 64]
checkpoints = ["k400", "ssv2"]

fig, axes = plt.subplots(len(frame_sizes), len(checkpoints), figsize=(14, 12), sharex=False, sharey=True)

for row, frames in enumerate(frame_sizes):
    for col, ckpt in enumerate(checkpoints):
        ax = axes[row][col]
        key = (ckpt, frames)
        if key in experiments:
            for strategy, color, ls in [("head", "#2196F3", "-"), ("all", "#F44336", "--")]:
                if strategy in experiments[key]:
                    df = experiments[key][strategy]
                    ax.plot(df["epoch"], df[METRIC], color=color, linestyle=ls,
                            label=f"unfreeze_{strategy}", linewidth=1.5, alpha=0.85)

        ax.set_title(f"{ckpt.upper()} — {frames}f", fontsize=11, fontweight="bold")
        ax.set_ylabel("AUC-ROC" if col == 0 else "")
        ax.set_xlabel("Epoch" if row == len(frame_sizes) - 1 else "")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)

fig.suptitle("TimeSformer — eval/auc_roc por Experimento", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = RESULTS_DIR / "auc_roc_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Salvo em {out}")
plt.show()
