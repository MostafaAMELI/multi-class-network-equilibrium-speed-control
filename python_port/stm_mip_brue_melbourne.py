#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np

SCENARIOS = {
    "UE": {"equilibrium": "ue", "speed_control": False, "include_time": False, "include_emission": False},
    "UE_SO": {"equilibrium": "ue", "speed_control": False, "include_time": True, "include_emission": False},
    "UE_SO_E": {"equilibrium": "ue", "speed_control": False, "include_time": True, "include_emission": True},
    "BRUE": {"equilibrium": "brue", "speed_control": False, "include_time": False, "include_emission": False},
    "BRUE_SO": {"equilibrium": "brue", "speed_control": False, "include_time": True, "include_emission": False},
    "BRUE_SO_E": {"equilibrium": "brue", "speed_control": False, "include_time": True, "include_emission": True},
    "SC_UE_SO_E": {"equilibrium": "ue", "speed_control": True, "include_time": True, "include_emission": True},
    "SC_BRUE_SO_E": {"equilibrium": "brue", "speed_control": True, "include_time": True, "include_emission": True},
}


def ef_speed(speed_kmh: float, vehicle_class: int) -> float:
    v = max(speed_kmh, 1e-3)
    if vehicle_class == 0:
        return 4.78e3 / v + 1.11e2 - 1.24 * v + 2.37e-2 * v * v
    return 3.67e3 / v + 5.34e2 - 7.90 * v + 5.43e-2 * v * v


def speed_bounds(network: Dict[str, np.ndarray], m: int, e: int, speed_control: bool) -> Tuple[float, float]:
    if speed_control:
        if m == 0:
            v_minus = float(network["spd_range_c1"][e]) * float(network["spdlimit_c"][e])
            v_plus = float(network["spd_range_c2"][e]) * float(network["spdlimit_c"][e])
        else:
            v_minus = float(network["spd_range_t1"][e]) * float(network["spdlimit_t"][e])
            v_plus = float(network["spd_range_t2"][e]) * float(network["spdlimit_t"][e])
    else:
        v0 = float(network["spdlimit_c"][e]) if m == 0 else float(network["spdlimit_t"][e])
        v_minus = 0.999 * v0
        v_plus = 1.001 * v0

    v_minus = max(v_minus, 1e-3)
    v_plus = max(v_plus, v_minus + 1e-4)
    return v_minus, v_plus


def _safe_int(s: str) -> Optional[int]:
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


def _load_nodes(node_csv: Path) -> Dict[int, Tuple[float, float]]:
    coords: Dict[int, Tuple[float, float]] = {}
    with node_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = _safe_int(row.get("node_id", ""))
            if nid is None:
                continue
            x = _safe_float(row.get("x_coord", ""), default=0.0)
            y = _safe_float(row.get("y_coord", ""), default=0.0)
            coords[nid] = (x, y)
    return coords


def _load_links(link_csv: Path, node_coords: Dict[int, Tuple[float, float]], truck_speed_factor: float) -> Dict[str, np.ndarray]:
    rows: List[Tuple[int, int, int, float, float, float]] = []

    with link_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lid = _safe_int(row.get("link_id", ""))
            u = _safe_int(row.get("from_node_id", ""))
            v = _safe_int(row.get("to_node_id", ""))
            if lid is None or u is None or v is None:
                continue

            cap = max(_safe_float(row.get("capacity", ""), default=1000.0), 1.0)
            car_speed = max(_safe_float(row.get("free_speed", ""), default=50.0), 1.0)
            truck_speed = max(min(car_speed * truck_speed_factor, car_speed), 1.0)

            length_km = _safe_float(row.get("length", ""), default=0.0)
            if length_km <= 0.0:
                if u in node_coords and v in node_coords:
                    x1, y1 = node_coords[u]
                    x2, y2 = node_coords[v]
                    length_km = max(float(np.hypot(x2 - x1, y2 - y1)) / 1000.0, 0.01)
                else:
                    length_km = 0.5

            rows.append((lid, u, v, length_km, car_speed, cap, truck_speed))

    if not rows:
        raise RuntimeError(f"No valid links read from {link_csv}")

    return {
        "id": np.array([r[0] for r in rows], dtype=int),
        "fromNode": np.array([r[1] for r in rows], dtype=int),
        "toNode": np.array([r[2] for r in rows], dtype=int),
        "length": np.array([r[3] for r in rows], dtype=float),
        "spdlimit_c": np.array([r[4] for r in rows], dtype=float),
        "capacity": np.array([r[5] for r in rows], dtype=float),
        "spdlimit_t": np.array([r[6] for r in rows], dtype=float),
    }


def _read_od_matrix(path: Path) -> List[Tuple[int, int, float]]:
    trips: List[Tuple[int, int, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        dests: List[Optional[int]] = []
        for h in header[1:]:
            if h.strip().lower() == "total":
                continue
            dests.append(_safe_int(h))

        for row in r:
            if not row:
                continue
            o = _safe_int(row[0])
            if o is None:
                continue
            values = row[1 : 1 + len(dests)]
            for d, val_s in zip(dests, values):
                if d is None:
                    continue
                val = _safe_float(val_s, default=0.0)
                if val > 0.0 and o != d:
                    trips.append((o, d, val))
    return trips


def _load_od_pairs(data_dir: Path, od_mode: str, od_slice: str) -> Dict[Tuple[int, int], float]:
    files = sorted(data_dir.glob("OD_matrix_*.csv"))
    if not files:
        raise RuntimeError(f"No OD matrices found in {data_dir}")

    if od_mode == "single":
        target = data_dir / f"OD_matrix_{od_slice}.csv"
        if not target.exists():
            raise RuntimeError(f"OD slice file not found: {target}")
        files = [target]

    demand: Dict[Tuple[int, int], float] = {}
    for fp in files:
        for o, d, val in _read_od_matrix(fp):
            demand[(o, d)] = demand.get((o, d), 0.0) + val
    return demand


def _build_adj(from_nodes: np.ndarray, to_nodes: np.ndarray) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for e in range(len(from_nodes)):
        u = int(from_nodes[e])
        adj.setdefault(u, []).append(e)
    return adj


def _dijkstra_edge_path(
    source: int,
    target: int,
    adj: Dict[int, List[int]],
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    edge_cost: np.ndarray,
    banned_nodes: Set[int],
    banned_edges: Set[int],
) -> Optional[Tuple[float, List[int], List[int]]]:
    pq: List[Tuple[float, int]] = [(0.0, source)]
    dist: Dict[int, float] = {source: 0.0}
    parent_node: Dict[int, int] = {}
    parent_edge: Dict[int, int] = {}

    while pq:
        g, u = heapq.heappop(pq)
        if g > dist.get(u, float("inf")):
            continue
        if u == target:
            break

        for e in adj.get(u, []):
            if e in banned_edges:
                continue
            v = int(to_nodes[e])
            if v in banned_nodes and v != target:
                continue

            ng = g + float(edge_cost[e])
            if ng + 1e-12 < dist.get(v, float("inf")):
                dist[v] = ng
                parent_node[v] = u
                parent_edge[v] = e
                heapq.heappush(pq, (ng, v))

    if target not in dist:
        return None

    edges_rev: List[int] = []
    nodes_rev: List[int] = [target]
    cur = target
    while cur != source:
        e = parent_edge[cur]
        p = parent_node[cur]
        edges_rev.append(e)
        nodes_rev.append(p)
        cur = p

    edges_rev.reverse()
    nodes_rev.reverse()
    return float(dist[target]), nodes_rev, edges_rev


def _k_shortest_edge_paths(
    source: int,
    target: int,
    k_paths: int,
    adj: Dict[int, List[int]],
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    edge_cost: np.ndarray,
) -> List[Tuple[float, List[int], List[int]]]:
    first = _dijkstra_edge_path(source, target, adj, from_nodes, to_nodes, edge_cost, set(), set())
    if first is None:
        return []

    A: List[Tuple[float, List[int], List[int]]] = [first]
    B: List[Tuple[float, int, List[int], List[int]]] = []
    candidate_seen: Set[Tuple[int, ...]] = {tuple(first[2])}
    serial = 0

    for _ in range(1, k_paths):
        prev_cost, prev_nodes, prev_edges = A[-1]
        del prev_cost

        for i in range(len(prev_nodes) - 1):
            root_nodes = prev_nodes[: i + 1]
            root_edges = prev_edges[:i]
            spur_node = root_nodes[-1]

            banned_edges: Set[int] = set()
            for _, a_nodes, a_edges in A:
                if len(a_nodes) > i and a_nodes[: i + 1] == root_nodes and len(a_edges) > i:
                    banned_edges.add(a_edges[i])

            banned_nodes = set(root_nodes[:-1])

            spur = _dijkstra_edge_path(
                spur_node,
                target,
                adj,
                from_nodes,
                to_nodes,
                edge_cost,
                banned_nodes,
                banned_edges,
            )
            if spur is None:
                continue

            spur_cost, spur_nodes, spur_edges = spur
            root_cost = 0.0
            for e in root_edges:
                root_cost += float(edge_cost[e])

            total_nodes = root_nodes[:-1] + spur_nodes
            total_edges = root_edges + spur_edges
            key = tuple(total_edges)
            if key in candidate_seen:
                continue

            candidate_seen.add(key)
            serial += 1
            heapq.heappush(B, (root_cost + spur_cost, serial, total_nodes, total_edges))

        if not B:
            break

        c = heapq.heappop(B)
        A.append((float(c[0]), c[2], c[3]))

    return A


def initialize_melbourne(
    data_dir: Path,
    od_mode: str,
    od_slice: str,
    top_n_od: int,
    k_paths: int,
    truck_share: float,
    truck_speed_factor: float,
    bpr_piece_l: int,
    bpr_piece_r: int,
    em_piece: int,
    bpr_max_x_ratio: float,
) -> Dict[str, object]:
    if not data_dir.exists():
        raise RuntimeError(f"Dataset directory does not exist: {data_dir}")

    node_coords = _load_nodes(data_dir / "node.csv")
    network = _load_links(data_dir / "link.csv", node_coords, truck_speed_factor=truck_speed_factor)

    # Speed-control bounds from the existing Tilburg implementation assumptions.
    nlink = len(network["id"])
    network["spd_range_c1"] = np.full(nlink, 0.70)
    network["spd_range_c2"] = np.full(nlink, 1.15)
    network["spd_range_t1"] = np.full(nlink, 0.80)
    network["spd_range_t2"] = np.full(nlink, 1.10)

    demand_map = _load_od_pairs(data_dir, od_mode=od_mode, od_slice=od_slice)

    from_nodes = network["fromNode"]
    to_nodes = network["toNode"]
    length = network["length"]
    spd_c = network["spdlimit_c"]
    spd_t = network["spdlimit_t"]

    adj = _build_adj(from_nodes, to_nodes)
    edge_cost_c = length / np.maximum(spd_c, 1e-3)
    edge_cost_t = length / np.maximum(spd_t, 1e-3)

    candidates = sorted(demand_map.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[Tuple[int, int, float]] = []
    car_paths_by_od: List[List[Tuple[float, List[int], List[int]]]] = []
    truck_paths_by_od: List[List[Tuple[float, List[int], List[int]]]] = []

    for (o, d), dem in candidates:
        if o not in adj:
            continue

        paths_c = _k_shortest_edge_paths(o, d, k_paths, adj, from_nodes, to_nodes, edge_cost_c)
        paths_t = _k_shortest_edge_paths(o, d, k_paths, adj, from_nodes, to_nodes, edge_cost_t)
        kp = min(len(paths_c), len(paths_t))
        if kp <= 0:
            continue

        selected.append((o, d, dem))
        car_paths_by_od.append(paths_c[:kp])
        truck_paths_by_od.append(paths_t[:kp])
        if len(selected) >= top_n_od:
            break

    if not selected:
        raise RuntimeError("No OD pairs with feasible paths were found.")

    no_class = 2
    no_od = len(selected)
    path_vector = np.array([min(len(car_paths_by_od[w]), len(truck_paths_by_od[w])) for w in range(no_od)], dtype=int)
    no_path = int(np.sum(path_vector))

    # Keep only links that appear in generated paths to keep the MILP tractable.
    used_edges: Set[int] = set()
    for w in range(no_od):
        for p in range(path_vector[w]):
            used_edges.update(car_paths_by_od[w][p][2])
            used_edges.update(truck_paths_by_od[w][p][2])
    if not used_edges:
        raise RuntimeError("No path edges selected for the reduced subnetwork.")

    keep_edges = np.array(sorted(used_edges), dtype=int)
    edge_remap = {int(old): int(new) for new, old in enumerate(keep_edges)}

    network = {
        key: np.asarray(val)[keep_edges]
        for key, val in network.items()
    }
    no_link = len(keep_edges)

    path_list = np.zeros((no_path, 2 + no_link, no_class), dtype=float)
    od_list = np.zeros((no_od, 4), dtype=float)

    row = 0
    for w, (o, d, dem) in enumerate(selected):
        d_truck = max(min(truck_share, 1.0), 0.0) * dem
        d_car = dem - d_truck
        od_list[w, :] = [o, d, d_car, d_truck]

        for p in range(path_vector[w]):
            for m in range(no_class):
                path_list[row + p, 0, m] = o
                path_list[row + p, 1, m] = d

            for e in car_paths_by_od[w][p][2]:
                path_list[row + p, 2 + edge_remap[int(e)], 0] = 1.0
            for e in truck_paths_by_od[w][p][2]:
                path_list[row + p, 2 + edge_remap[int(e)], 1] = 1.0

        row += path_vector[w]

    # Link-specific upper bound on aggregate PCE flow implied by selected OD demands
    # and available candidate paths. This keeps BPR linearization domain feasible/tight.
    pce = np.array([1.0, 2.5], dtype=float)
    xagg_ub = np.zeros(no_link, dtype=float)
    row = 0
    for w in range(no_od):
        d_class = np.array([od_list[w, 2], od_list[w, 3]], dtype=float)
        pw = int(path_vector[w])
        for e in range(no_link):
            for m in range(no_class):
                # If any candidate path for OD w and class m uses link e,
                # that class demand may contribute to link e in the worst case.
                if np.any(path_list[row : row + pw, 2 + e, m] > 0.5):
                    xagg_ub[e] += pce[m] * d_class[m]
        row += pw

    return {
        "BPR_piece_l": int(bpr_piece_l),
        "BPR_piece_r": int(bpr_piece_r),
        "BPR_piece": int(bpr_piece_l + bpr_piece_r),
        "em_piece": int(em_piece),
        "bpr_max_x_ratio": float(max(1.0, bpr_max_x_ratio)),
        "pce": np.array([1.0, 2.5]),
        "xi": np.array([9.0, 38.0]),
        "network": network,
        "no_link": no_link,
        "no_class": no_class,
        "no_OD": no_od,
        "no_path": no_path,
        "OD_list": od_list,
        "path_list": path_list,
        "path_vector": path_vector,
        "xagg_ub": xagg_ub,
        "selected_od": selected,
        "k_paths": k_paths,
        "od_mode": od_mode,
        "od_slice": od_slice,
        "top_n_od": top_n_od,
        "truck_share": truck_share,
    }


def build_model(
    nw: Dict[str, object],
    equilibrium: str,
    speed_control: bool,
    include_time: bool,
    include_emission: bool,
    epsilon: float,
    theta: float,
    timelimit: float,
    env: Optional[gp.Env] = None,
) -> Tuple[gp.Model, Dict[str, object]]:
    no_class = int(nw["no_class"])
    no_path = int(nw["no_path"])
    no_od = int(nw["no_OD"])
    no_link = int(nw["no_link"])
    bpr_piece = int(nw["BPR_piece"])
    bpr_max_x_ratio = float(nw.get("bpr_max_x_ratio", float(bpr_piece) / float(nw["BPR_piece_l"])))
    pce = np.asarray(nw["pce"], dtype=float)
    xi = np.asarray(nw["xi"], dtype=float)
    path_list = np.asarray(nw["path_list"], dtype=float)
    od_list = np.asarray(nw["OD_list"], dtype=float)
    network = nw["network"]
    xagg_ub = np.asarray(nw.get("xagg_ub", np.array([])), dtype=float)

    M = 500000.0
    Mflow = 300000.0

    od_to_paths: Dict[Tuple[int, int], List[int]] = {}
    path_od_idx: Dict[Tuple[int, int], int] = {}
    for p in range(no_path):
        od_key = (int(path_list[p, 0, 0]), int(path_list[p, 1, 0]))
        od_to_paths.setdefault(od_key, []).append(p)
    for w in range(no_od):
        path_od_idx[(int(od_list[w, 0]), int(od_list[w, 1]))] = w

    model = gp.Model("stm_mip_brue_melbourne", env=env) if env is not None else gp.Model("stm_mip_brue_melbourne")
    model.Params.TimeLimit = timelimit

    f = model.addVars(no_class, no_path, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    a = model.addVars(no_class, no_path, vtype=GRB.BINARY, name="a")
    cstar = model.addVars(no_class, no_od, lb=0.0, vtype=GRB.CONTINUOUS, name="cstar")
    x_link = model.addVars(no_class, no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="x_link")
    x_agg = model.addVars(no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="x_agg")
    c_link = model.addVars(no_class, no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="c_link")
    c_path = model.addVars(no_class, no_path, lb=0.0, vtype=GRB.CONTINUOUS, name="c_path")

    lamb = model.addVars(no_link, bpr_piece + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="bpr_lam")
    bpr_val = model.addVars(no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="bpr_val")

    em_piece = int(nw["em_piece"])
    eta = model.addVars(no_class, no_link, em_piece, lb=0.0, vtype=GRB.CONTINUOUS, name="em_eta")
    ebar = model.addVars(no_class, no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="e_bar")

    for m in range(no_class):
        for w in range(no_od):
            od_key = (int(od_list[w, 0]), int(od_list[w, 1]))
            model.addConstr(
                gp.quicksum(f[m, p] for p in od_to_paths.get(od_key, [])) == float(od_list[w, 2 + m]),
                name=f"flow_{m}_{w}",
            )

    for m in range(no_class):
        for e in range(no_link):
            model.addConstr(
                x_link[m, e] == gp.quicksum(float(path_list[p, 2 + e, m]) * f[m, p] for p in range(no_path)),
                name=f"xlink_{m}_{e}",
            )

    for e in range(no_link):
        model.addConstr(
            x_agg[e] == gp.quicksum(float(pce[m]) * x_link[m, e] for m in range(no_class)),
            name=f"xagg_{e}",
        )

    x_ratio_max_e = np.full(no_link, float(bpr_max_x_ratio), dtype=float)
    x_max_e = np.zeros(no_link, dtype=float)

    for e in range(no_link):
        cap = float(network["capacity"][e])
        length = float(network["length"][e])
        if xagg_ub.size == no_link:
            x_max = max(float(xagg_ub[e]) * 1.05, cap * 1.01)
        else:
            x_max = cap * bpr_max_x_ratio
        x_max_e[e] = x_max
        x_ratio_max_e[e] = max(x_max / max(cap, 1e-6), 1.0)

        x_bp = [x_max * l / float(bpr_piece) for l in range(bpr_piece + 1)]
        y_bp = [length * (1.0 + 0.15 * (xx / cap) ** 4) for xx in x_bp]

        model.addConstr(gp.quicksum(lamb[e, l] for l in range(bpr_piece + 1)) == 1.0, name=f"lam_sum_{e}")
        model.addConstr(gp.quicksum(x_bp[l] * lamb[e, l] for l in range(bpr_piece + 1)) == x_agg[e], name=f"lam_x_{e}")
        model.addConstr(gp.quicksum(y_bp[l] * lamb[e, l] for l in range(bpr_piece + 1)) == bpr_val[e], name=f"lam_y_{e}")
        model.addSOS(GRB.SOS_TYPE2, [lamb[e, l] for l in range(bpr_piece + 1)], x_bp)

    for m in range(no_class):
        for e in range(no_link):
            v_minus, v_plus = speed_bounds(network, m, e, speed_control)
            model.addConstr(c_link[m, e] * v_minus <= bpr_val[e], name=f"sc_lb_{m}_{e}")
            model.addConstr(c_link[m, e] * v_plus >= bpr_val[e], name=f"sc_ub_{m}_{e}")

    for m in range(no_class):
        for p in range(no_path):
            model.addConstr(
                c_path[m, p] == gp.quicksum(float(path_list[p, 2 + e, m]) * c_link[m, e] for e in range(no_link)),
                name=f"cpath_{m}_{p}",
            )

    for m in range(no_class):
        for p in range(no_path):
            od_key = (int(path_list[p, 0, m]), int(path_list[p, 1, m]))
            w = path_od_idx[od_key]

            model.addConstr(c_path[m, p] - cstar[m, w] >= 0.0, name=f"eq_lb_{m}_{p}")
            if equilibrium == "ue":
                model.addConstr(c_path[m, p] - cstar[m, w] <= M * (1.0 - a[m, p]), name=f"ue_ub_{m}_{p}")
            else:
                model.addConstr(
                    c_path[m, p] - (1.0 + epsilon) * cstar[m, w] <= M * (1.0 - a[m, p]),
                    name=f"brue_ub_{m}_{p}",
                )

            # Large-scale networks may have many equal-cost alternatives.
            # Enforcing a strict gap for every unused path can make the model infeasible.
            # Keep the implication f>0 => a=1 only; unused paths remain unconstrained above cstar.
            model.addConstr(f[m, p] - Mflow * a[m, p] <= 0.0, name=f"alink2_{m}_{p}")

    for m in range(no_class):
        for e in range(no_link):
            v_minus, v_plus = speed_bounds(network, m, e, speed_control)
            length = float(network["length"][e])
            x_ratio_max = float(x_ratio_max_e[e])
            bpr_factor_max = 1.0 + 0.15 * (x_ratio_max ** 4)
            c_min = length / v_plus
            c_max = max((length / v_minus) * bpr_factor_max * 1.05, c_min * 2.0)
            c_bp = np.linspace(c_min, c_max, em_piece)
            e_bp = []
            for cc in c_bp:
                speed = length / max(cc, 1e-6)
                ef = ef_speed(speed, m)
                e_bp.append((ef * length) / 1e6)

            model.addConstr(gp.quicksum(eta[m, e, k] for k in range(em_piece)) == 1.0, name=f"eta_sum_{m}_{e}")
            model.addConstr(
                gp.quicksum(float(c_bp[k]) * eta[m, e, k] for k in range(em_piece)) == c_link[m, e],
                name=f"eta_c_{m}_{e}",
            )
            model.addConstr(
                gp.quicksum(float(e_bp[k]) * eta[m, e, k] for k in range(em_piece)) == ebar[m, e],
                name=f"eta_e_{m}_{e}",
            )
            model.addSOS(GRB.SOS_TYPE2, [eta[m, e, k] for k in range(em_piece)], list(c_bp))

    obj = gp.LinExpr()

    if include_time:
        f_ub = np.zeros((no_class, no_path), dtype=float)
        cp_ub = np.zeros((no_class, no_path), dtype=float)
        for m in range(no_class):
            for p in range(no_path):
                od_key = (int(path_list[p, 0, m]), int(path_list[p, 1, m]))
                w = path_od_idx[od_key]
                f_ub[m, p] = float(od_list[w, 2 + m])

                ub = 0.0
                for e in range(no_link):
                    if float(path_list[p, 2 + e, m]) <= 0.5:
                        continue
                    v_minus, _ = speed_bounds(network, m, e, speed_control)
                    bpr_factor_e = 1.0 + 0.15 * (float(x_ratio_max_e[e]) ** 4)
                    ub += float(network["length"][e]) / max(v_minus, 1e-3) * bpr_factor_e * 1.1
                cp_ub[m, p] = max(ub, 1e-3)

        z_time = model.addVars(no_class, no_path, lb=0.0, vtype=GRB.CONTINUOUS, name="z_time")
        for m in range(no_class):
            for p in range(no_path):
                ux = float(f_ub[m, p])
                uy = float(cp_ub[m, p])
                model.addConstr(z_time[m, p] <= ux * c_path[m, p], name=f"mcc_t_ub1_{m}_{p}")
                model.addConstr(z_time[m, p] <= uy * f[m, p], name=f"mcc_t_ub2_{m}_{p}")
                model.addConstr(
                    z_time[m, p] >= uy * f[m, p] + ux * c_path[m, p] - ux * uy,
                    name=f"mcc_t_lb_{m}_{p}",
                )
                obj += float(xi[m]) * z_time[m, p]

    if include_emission:
        class_demand = np.array([float(np.sum(od_list[:, 2 + m])) for m in range(no_class)], dtype=float)
        ebar_ub = np.zeros((no_class, no_link), dtype=float)
        for m in range(no_class):
            for e in range(no_link):
                v_minus, v_plus = speed_bounds(network, m, e, speed_control)
                length = float(network["length"][e])
                bpr_factor_max = 1.0 + 0.15 * (float(x_ratio_max_e[e]) ** 4)
                c_min = length / v_plus
                c_max = max((length / v_minus) * bpr_factor_max * 1.05, c_min * 2.0)
                speed_min = length / max(c_max, 1e-6)
                speed_max = length / max(c_min, 1e-6)
                ef1 = ef_speed(speed_min, m)
                ef2 = ef_speed(speed_max, m)
                ebar_ub[m, e] = max((ef1 * length) / 1e6, (ef2 * length) / 1e6, 1e-6)

        z_em = model.addVars(no_class, no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="z_em")
        for m in range(no_class):
            ux = float(class_demand[m])
            for e in range(no_link):
                uy = float(ebar_ub[m, e])
                model.addConstr(z_em[m, e] <= ux * ebar[m, e], name=f"mcc_e_ub1_{m}_{e}")
                model.addConstr(z_em[m, e] <= uy * x_link[m, e], name=f"mcc_e_ub2_{m}_{e}")
                model.addConstr(
                    z_em[m, e] >= uy * x_link[m, e] + ux * ebar[m, e] - ux * uy,
                    name=f"mcc_e_lb_{m}_{e}",
                )
                obj += float(theta) * z_em[m, e]

    model.setObjective(obj, GRB.MINIMIZE)

    return model, {
        "f": f,
        "x_link": x_link,
        "c_path": c_path,
        "ebar": ebar,
    }


def solve_with_nw(
    nw: Dict[str, object],
    scenario_name: str,
    equilibrium: str,
    speed_control: bool,
    include_time: bool,
    include_emission: bool,
    epsilon: float,
    theta: float,
    timelimit: float,
    output: Path,
    iis_dir: Optional[Path] = None,
    env: Optional[gp.Env] = None,
) -> Dict[str, object]:
    last_err: Optional[Exception] = None
    model = None
    h = None
    for attempt in range(3):
        try:
            model, h = build_model(
                nw=nw,
                equilibrium=equilibrium,
                speed_control=speed_control,
                include_time=include_time,
                include_emission=include_emission,
                epsilon=epsilon,
                theta=theta,
                timelimit=timelimit,
                env=env,
            )
            model.optimize()
            break
        except gp.GurobiError as e:
            last_err = e
            msg = str(e).lower()
            if "could not resolve host" in msg and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    if model is None or h is None:
        raise RuntimeError(f"Model creation/optimization failed: {last_err}")

    if model.SolCount == 0:
        if model.Status == GRB.INFEASIBLE and iis_dir is not None:
            iis_dir.mkdir(parents=True, exist_ok=True)
            model.computeIIS()
            iis_file = iis_dir / f"iis_{scenario_name.lower()}.ilp"
            model.write(str(iis_file))
        raise RuntimeError(f"No solution found. Status={model.Status}")

    no_class = int(nw["no_class"])
    no_path = int(nw["no_path"])
    no_link = int(nw["no_link"])
    xi = np.asarray(nw["xi"], dtype=float)

    f_val = np.array([[h["f"][m, p].X for p in range(no_path)] for m in range(no_class)], dtype=float)
    cp_val = np.array([[h["c_path"][m, p].X for p in range(no_path)] for m in range(no_class)], dtype=float)
    xlink_val = np.array([[h["x_link"][m, e].X for e in range(no_link)] for m in range(no_class)], dtype=float)
    ebar_val = np.array([[h["ebar"][m, e].X for e in range(no_link)] for m in range(no_class)], dtype=float)

    j1 = float(np.sum((xi[:, None] * f_val) * cp_val))
    j2 = float(theta * np.sum(ebar_val * xlink_val))
    tt_car_h = float(np.sum(f_val[0, :] * cp_val[0, :]))
    tt_truck_h = float(np.sum(f_val[1, :] * cp_val[1, :]))
    em_car_ton = float(np.sum(ebar_val[0, :] * xlink_val[0, :]))
    em_truck_ton = float(np.sum(ebar_val[1, :] * xlink_val[1, :]))
    time_cost_car = float(xi[0] * tt_car_h)
    time_cost_truck = float(xi[1] * tt_truck_h)

    summary = {
        "status": int(model.Status),
        "objective": float(model.ObjVal),
        "J1_time_cost": j1,
        "J2_emission_cost": j2,
        "tt_car_h": tt_car_h,
        "tt_truck_h": tt_truck_h,
        "tt_total_h": tt_car_h + tt_truck_h,
        "em_car_ton": em_car_ton,
        "em_truck_ton": em_truck_ton,
        "em_total_ton": em_car_ton + em_truck_ton,
        "time_cost_car_eur": time_cost_car,
        "time_cost_truck_eur": time_cost_truck,
        "time_cost_total_eur": time_cost_car + time_cost_truck,
        "emission_cost_total_eur": j2,
        "scenario": scenario_name,
        "equilibrium": equilibrium,
        "speed_control": bool(speed_control),
        "include_time": bool(include_time),
        "include_emission": bool(include_emission),
        "epsilon": float(epsilon),
        "theta": float(theta),
        "car_flow_total": float(np.sum(xlink_val[0, :])),
        "truck_flow_total": float(np.sum(xlink_val[1, :])),
        "dataset": "melbourne_gmns_4541153",
        "od_mode": nw["od_mode"],
        "od_slice": nw["od_slice"],
        "top_n_od": int(nw["top_n_od"]),
        "k_paths": int(nw["k_paths"]),
        "truck_share": float(nw["truck_share"]),
        "no_link_reduced": int(nw["no_link"]),
        "bpr_piece_l": int(nw["BPR_piece_l"]),
        "bpr_piece_r": int(nw["BPR_piece_r"]),
        "bpr_max_x_ratio": float(nw["bpr_max_x_ratio"]),
        "em_piece": int(nw["em_piece"]),
        "selected_od": [
            {"origin": int(o), "destination": int(d), "demand_total": float(v)}
            for o, d, v in nw["selected_od"]
        ],
        "link_ids": [int(v) for v in np.asarray(nw["network"]["id"]).tolist()],
        "link_flow_car": [float(v) for v in xlink_val[0, :].tolist()],
        "link_flow_truck": [float(v) for v in xlink_val[1, :].tolist()],
        "link_flow_total_pce": [float(v) for v in (xlink_val[0, :] + 2.5 * xlink_val[1, :]).tolist()],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melbourne GMNS testcase for UE/BRUE and speed-control scenarios"
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="SC_BRUE_SO_E")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--od-mode", choices=["single", "aggregate"], default="single")
    parser.add_argument("--od-slice", type=str, default="800_815")
    parser.add_argument("--top-od", type=int, default=8)
    parser.add_argument("--k-paths", type=int, default=2)
    parser.add_argument("--truck-share", type=float, default=0.15)
    parser.add_argument("--truck-speed-factor", type=float, default=0.90)
    parser.add_argument("--bpr-piece-l", type=int, default=1)
    parser.add_argument("--bpr-piece-r", type=int, default=1)
    parser.add_argument(
        "--bpr-max-x-ratio",
        type=float,
        default=None,
        help="Max x/capacity ratio covered by BPR piecewise linearization (for high-demand feasibility).",
    )
    parser.add_argument("--em-piece", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--theta", type=float, default=70.0)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "melbourne" / "summary_melbourne.json",
    )
    parser.add_argument(
        "--iis-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "melbourne" / "iis",
        help="Directory to write IIS files when a scenario is infeasible.",
    )
    args = parser.parse_args()

    # Auto-pick a local WLS license file if env var is not already set.
    if "GRB_LICENSE_FILE" not in os.environ:
        default_lic = Path("/Users/ameli/Downloads/gurobi.lic")
        if default_lic.exists():
            os.environ["GRB_LICENSE_FILE"] = str(default_lic)

    data_dir = args.data_dir or (args.project_root / "testcases" / "4541153")
    bpr_piece_l = max(1, args.bpr_piece_l)
    bpr_piece_r = max(1, args.bpr_piece_r)
    bpr_max_x_ratio = args.bpr_max_x_ratio
    if bpr_max_x_ratio is None:
        bpr_max_x_ratio = float(bpr_piece_l + bpr_piece_r) / float(bpr_piece_l)

    nw = initialize_melbourne(
        data_dir=data_dir,
        od_mode=args.od_mode,
        od_slice=args.od_slice,
        top_n_od=max(1, args.top_od),
        k_paths=max(1, args.k_paths),
        truck_share=args.truck_share,
        truck_speed_factor=args.truck_speed_factor,
        bpr_piece_l=bpr_piece_l,
        bpr_piece_r=bpr_piece_r,
        em_piece=max(3, args.em_piece),
        bpr_max_x_ratio=bpr_max_x_ratio,
    )

    shared_env: Optional[gp.Env] = None
    try:
        # Reuse a single environment for the full bundle to avoid repeated WLS handshakes.
        shared_env = gp.Env()
    except gp.GurobiError:
        shared_env = None

    if args.all_scenarios:
        bundle: Dict[str, Dict[str, object]] = {}
        for name, cfg in SCENARIOS.items():
            out_i = args.output.parent / f"summary_melbourne_{name.lower()}.json"
            s = solve_with_nw(
                nw=nw,
                scenario_name=name,
                equilibrium=cfg["equilibrium"],
                speed_control=cfg["speed_control"],
                include_time=cfg["include_time"],
                include_emission=cfg["include_emission"],
                epsilon=args.epsilon,
                theta=args.theta,
                timelimit=args.time_limit,
                output=out_i,
                iis_dir=args.iis_dir,
                env=shared_env,
            )
            bundle[name] = s
            print(f"{name}: status={s['status']} objective={s['objective']:.6f}")

        combined = args.output.parent / "summary_melbourne_all.json"
        with combined.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        print(f"saved={combined}")
        return

    cfg = SCENARIOS[args.scenario]
    s = solve_with_nw(
        nw=nw,
        scenario_name=args.scenario,
        equilibrium=cfg["equilibrium"],
        speed_control=cfg["speed_control"],
        include_time=cfg["include_time"],
        include_emission=cfg["include_emission"],
        epsilon=args.epsilon,
        theta=args.theta,
        timelimit=args.time_limit,
        output=args.output,
        iis_dir=args.iis_dir,
        env=shared_env,
    )
    print(f"scenario={s['scenario']}")
    print(f"status={s['status']}")
    print(f"objective={s['objective']:.6f}")
    print(f"J1_time_cost={s['J1_time_cost']:.6f}")
    print(f"J2_emission_cost={s['J2_emission_cost']:.6f}")
    print(f"car_flow_total={s['car_flow_total']:.6f}")
    print(f"truck_flow_total={s['truck_flow_total']:.6f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
