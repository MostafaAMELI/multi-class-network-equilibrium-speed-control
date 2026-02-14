#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_int(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _safe_float(s: str, default: float = 0.0) -> float:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def load_nodes(node_csv: Path) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    with node_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = _safe_int(row.get("node_id", ""))
            if nid is None:
                continue
            x = _safe_float(row.get("x_coord", ""), 0.0)
            y = _safe_float(row.get("y_coord", ""), 0.0)
            out[nid] = (x, y)
    return out


def load_links(link_csv: Path) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    with link_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lid = _safe_int(row.get("link_id", ""))
            u = _safe_int(row.get("from_node_id", ""))
            v = _safe_int(row.get("to_node_id", ""))
            if lid is None or u is None or v is None:
                continue
            out[lid] = (u, v)
    return out


def draw_scenario(
    scenario: str,
    sdata: dict,
    nodes: Dict[int, Tuple[float, float]],
    links: Dict[int, Tuple[int, int]],
    metric: str,
    out_path: Path,
    annotate: bool,
    cmap_name: str,
    bg_color: str,
    line_min: float,
    line_max: float,
    label_size: float,
) -> None:
    metric_key = {
        "car": "link_flow_car",
        "truck": "link_flow_truck",
        "total_pce": "link_flow_total_pce",
    }[metric]

    lids = [int(x) for x in sdata.get("link_ids", [])]
    vals = [float(x) for x in sdata.get(metric_key, [])]
    if not lids or not vals or len(lids) != len(vals):
        raise RuntimeError(f"Scenario {scenario} has no usable link-flow arrays ({metric_key}).")

    flow = {lid: val for lid, val in zip(lids, vals)}
    arr = np.array(vals, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    denom = max(vmax - vmin, 1e-9)

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.get_cmap(cmap_name)
    ax.set_facecolor(bg_color)

    for lid in lids:
        if lid not in links:
            continue
        u, v = links[lid]
        if u not in nodes or v not in nodes:
            continue
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        fv = flow[lid]
        t = (fv - vmin) / denom
        color = cmap(t)
        lw = line_min + (line_max - line_min) * t
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.95)

        if annotate:
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            ax.text(mx, my, f"{fv:.0f}", fontsize=label_size, color="#202020", ha="center", va="center")

    ax.set_title(f"Melbourne Network Link Flows - {scenario} ({metric})")
    ax.set_xlabel("X (UTM)")
    ax.set_ylabel("Y (UTM)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"{metric} flow")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Melbourne network with per-link flow values for scenarios")
    p.add_argument("--summary", type=Path, default=Path(__file__).resolve().parent / "output" / "melbourne" / "summary_melbourne_all.json")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "testcases" / "4541153")
    p.add_argument("--scenarios", nargs="+", default=["UE_SO", "UE_SO_E"])
    p.add_argument("--metric", choices=["car", "truck", "total_pce"], default="total_pce")
    p.add_argument("--annotate", action="store_true")
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument("--bg-color", type=str, default="#d9d9d9")
    p.add_argument("--line-min", type=float, default=1.0)
    p.add_argument("--line-max", type=float, default=8.0)
    p.add_argument("--label-size", type=float, default=5.0)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "melbourne")
    args = p.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    nodes = load_nodes(args.data_dir / "node.csv")
    links = load_links(args.data_dir / "link.csv")

    for s in args.scenarios:
        if s not in data:
            raise RuntimeError(f"Scenario not found in summary: {s}")
        out = args.out_dir / f"network_linkflow_{s.lower()}_{args.metric}.png"
        draw_scenario(
            s,
            data[s],
            nodes,
            links,
            args.metric,
            out,
            args.annotate,
            args.cmap,
            args.bg_color,
            args.line_min,
            args.line_max,
            args.label_size,
        )
        print(f"saved={out}")


if __name__ == "__main__":
    main()
