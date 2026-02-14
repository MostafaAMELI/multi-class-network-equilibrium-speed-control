from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _edge_geometry(node: Dict[str, np.ndarray], network: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = node["xco"]
    y = node["yco"]
    start = network["fromNode"].astype(int) - 1
    end = network["toNode"].astype(int) - 1
    up_x = x[start]
    up_y = y[start]
    down_x = x[end]
    down_y = y[end]
    return up_x, up_y, down_x, down_y


def plot_network(node: Dict[str, np.ndarray], network: Dict[str, np.ndarray], title: str, out_file: Path, show_label: bool = True) -> None:
    x = node["xco"]
    y = node["yco"]
    up_x, up_y, down_x, down_y = _edge_geometry(node, network)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(x, y, ".", color="black", markersize=6)
    for i in range(len(up_x)):
        ax.plot([up_x[i], down_x[i]], [up_y[i], down_y[i]], color="black", linewidth=0.7, alpha=0.7)

    if show_label:
        for i in range(len(x)):
            ax.text(x[i], y[i], str(i + 1), fontsize=8, color="black")
        for i in range(len(up_x)):
            cx = 0.5 * (up_x[i] + down_x[i])
            cy = 0.5 * (up_y[i] + down_y[i])
            ax.text(cx, cy, str(i + 1), fontsize=7, color="#555555")

    ax.set_title(title)
    ax.set_axis_off()
    _apply_margin(ax, x, y)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_loaded_links(
    node: Dict[str, np.ndarray],
    network: Dict[str, np.ndarray],
    load: np.ndarray,
    title: str,
    out_file: Path,
    show_labels: bool = True,
) -> None:
    x = node["xco"]
    y = node["yco"]
    up_x, up_y, down_x, down_y = _edge_geometry(node, network)

    load = np.asarray(load, dtype=float)
    vmin = 0.0
    vmax = max(float(np.max(load)), 1e-9)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("hsv")

    seg = [((up_x[i], up_y[i]), (down_x[i], down_y[i])) for i in range(len(load))]
    width = 1.0 + 4.0 * (load / vmax if vmax > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(x, y, ".", color="black", markersize=5)
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=width, alpha=0.85)
    lc.set_array(load)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)

    if show_labels:
        for i in range(len(load)):
            cx = 0.5 * (up_x[i] + down_x[i])
            cy = 0.5 * (up_y[i] + down_y[i])
            ax.text(cx, cy, f"{int(round(load[i]))}", fontsize=7, color="black")

    ax.set_title(title)
    ax.set_axis_off()
    _apply_margin(ax, x, y)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_links_speed_limit_change(
    node: Dict[str, np.ndarray],
    network: Dict[str, np.ndarray],
    delta_pct: np.ndarray,
    title: str,
    out_file: Path,
    show_labels: bool = True,
) -> None:
    x = node["xco"]
    y = node["yco"]
    up_x, up_y, down_x, down_y = _edge_geometry(node, network)
    delta_pct = np.asarray(delta_pct, dtype=float)

    m = float(np.max(np.abs(delta_pct))) if len(delta_pct) else 1.0
    m = max(m, 1e-9)
    norm = Normalize(vmin=-m, vmax=m)
    cmap = plt.get_cmap("bwr")
    seg = [((up_x[i], up_y[i]), (down_x[i], down_y[i])) for i in range(len(delta_pct))]
    width = 1.0 + 4.0 * (np.abs(delta_pct) / m)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(x, y, ".", color="black", markersize=5)
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=width, alpha=0.9)
    lc.set_array(delta_pct)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)

    if show_labels:
        for i in range(len(delta_pct)):
            cx = 0.5 * (up_x[i] + down_x[i])
            cy = 0.5 * (up_y[i] + down_y[i])
            color = "blue" if delta_pct[i] > 0 else ("red" if delta_pct[i] < 0 else "black")
            ax.text(cx, cy, f"{int(round(delta_pct[i]))}", fontsize=7, color=color)

    ax.set_title(title)
    ax.set_axis_off()
    _apply_margin(ax, x, y)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_computation_time(out_file: Path) -> None:
    xsos = np.array([20, 40, 80, 120, 140, 158, 170, 180, 216, 219], dtype=float)
    ysos = np.array([0.11, 1.83, 3.92, 30.45, 66.91, 257.31, 337.39, 675, 1354, 12410], dtype=float)
    xori = np.array([20, 40, 60, 80, 100, 126, 131], dtype=float)
    yori = np.array([0.187, 6.08, 20.67, 73.84, 374.34, 4615, 24231], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(xori, np.log10(yori), "b-*", linewidth=1.5, label="MILP")
    ax.plot(xsos, np.log10(ysos), "r-x", linewidth=1.5, label="MILP-SOS")
    ax.set_xlabel("Number of OD pairs")
    ax.set_ylabel("Computation time (log10)")
    ax.legend(loc="upper left")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_link_capacity(node: Dict[str, np.ndarray], network: Dict[str, np.ndarray], out_file: Path) -> None:
    plot_loaded_links(
        node=node,
        network=network,
        load=np.round(network["capacity"]),
        title="Link Capacity",
        out_file=out_file,
        show_labels=True,
    )


def _apply_margin(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    marg_x = 0.1 * (float(np.max(x)) - float(np.min(x))) + 1e-6
    marg_y = 0.1 * (float(np.max(y)) - float(np.min(y))) + 1e-6
    ax.set_xlim(float(np.min(x)) - marg_x, float(np.max(x)) + marg_x)
    ax.set_ylim(float(np.min(y)) - marg_y, float(np.max(y)) + marg_y)

