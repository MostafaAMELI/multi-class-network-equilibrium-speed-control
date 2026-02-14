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
    p = argparse.ArgumentParser(description="Plot UE SO vs UE SO-E comparison from Melbourne summary JSON")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "melbourne" / "summary_melbourne_all.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "melbourne" / "ue_so_vs_ue_so_e.png",
    )
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if "UE_SO" not in data or "UE_SO_E" not in data:
        raise RuntimeError("Input summary must contain both UE_SO and UE_SO_E scenarios.")

    ue_so = data["UE_SO"]
    ue_so_e = data["UE_SO_E"]

    labels = ["Objective", "J1 (Time)", "J2 (Emission)"]
    v1 = [float(ue_so.get("objective", 0.0)), float(ue_so.get("J1_time_cost", 0.0)), float(ue_so.get("J2_emission_cost", 0.0))]
    v2 = [float(ue_so_e.get("objective", 0.0)), float(ue_so_e.get("J1_time_cost", 0.0)), float(ue_so_e.get("J2_emission_cost", 0.0))]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, v1, w, label="UE SO", color="#4C78A8")
    ax.bar(x + w / 2, v2, w, label="UE SO-E", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost (model units)")
    ax.set_title("Melbourne Testcase: UE SO vs UE SO-E")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
