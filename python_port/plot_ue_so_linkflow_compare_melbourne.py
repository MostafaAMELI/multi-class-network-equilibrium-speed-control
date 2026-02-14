#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Plot UE SO vs UE SO-E link-flow comparison (Melbourne)")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "melbourne" / "summary_melbourne_all.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "melbourne" / "ue_so_vs_ue_so_e_linkflow.png",
    )
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    ue_so = data.get("UE_SO")
    ue_so_e = data.get("UE_SO_E")
    if ue_so is None or ue_so_e is None:
        raise RuntimeError("Input summary must contain UE_SO and UE_SO_E.")

    f1 = np.array(ue_so.get("link_flow_total_pce", []), dtype=float)
    f2 = np.array(ue_so_e.get("link_flow_total_pce", []), dtype=float)
    lid1 = np.array(ue_so.get("link_ids", []), dtype=int)
    lid2 = np.array(ue_so_e.get("link_ids", []), dtype=int)

    if f1.size == 0 or f2.size == 0 or f1.size != f2.size or lid1.size != lid2.size:
        raise RuntimeError("Missing or mismatched link-flow arrays. Re-run scenarios with updated script.")

    # Align by link id (robust even if ordering changes).
    map1 = {int(l): float(v) for l, v in zip(lid1, f1)}
    map2 = {int(l): float(v) for l, v in zip(lid2, f2)}
    common = sorted(set(map1).intersection(map2))
    x = np.array([map1[l] for l in common], dtype=float)
    y = np.array([map2[l] for l in common], dtype=float)

    diff = y - x
    idx = np.argsort(np.abs(diff))[::-1][:10]
    top_links = [common[i] for i in idx]
    top_diff = diff[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    lo = float(min(np.min(x), np.min(y))) if x.size else 0.0
    hi = float(max(np.max(x), np.max(y))) if x.size else 1.0
    ax.scatter(x, y, s=22, alpha=0.8, color="#1f77b4")
    ax.plot([lo, hi], [lo, hi], "--", color="#444444", linewidth=1)
    ax.set_xlabel("UE SO link flow (PCE)")
    ax.set_ylabel("UE SO-E link flow (PCE)")
    ax.set_title("Link Flow Scatter")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax2 = axes[1]
    labels = [str(l) for l in top_links]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_diff]
    ax2.bar(np.arange(len(top_diff)), top_diff, color=colors)
    ax2.axhline(0.0, color="#444444", linewidth=1)
    ax2.set_xticks(np.arange(len(top_diff)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("UE SO-E - UE SO (PCE)")
    ax2.set_title("Top 10 Absolute Link-Flow Differences")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Melbourne Testcase: UE SO vs UE SO-E Link Flows", fontsize=12)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
