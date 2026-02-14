#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from reporting import calc_agap
from visualization import (
    plot_computation_time,
    plot_link_capacity,
    plot_links_speed_limit_change,
    plot_loaded_links,
    plot_network,
)


INF = float("inf")


def read_table(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        rows: List[List[float]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    arr = np.asarray(rows, dtype=float)
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(header):
        clean = "".join(ch for ch in name if ch.isalnum() or ch == "_")
        out[clean] = arr[:, i]
    return out


def dijkstra(cost: np.ndarray, source: int, dest: int) -> Tuple[List[int], float]:
    n = cost.shape[0]
    s = source - 1
    d = dest - 1
    visited = np.zeros(n, dtype=bool)
    dist = np.full(n, INF)
    parent = np.full(n, -1, dtype=int)
    dist[s] = 0.0

    for _ in range(n - 1):
        masked = np.where(visited, INF, dist)
        u = int(np.argmin(masked))
        if masked[u] == INF:
            break
        visited[u] = True
        for v in range(n):
            alt = dist[u] + cost[u, v]
            if alt < dist[v]:
                dist[v] = alt
                parent[v] = u

    if parent[d] == -1 and d != s:
        return [], INF

    path = [d]
    t = d
    while t != s:
        t = parent[t]
        if t == -1:
            return [], INF
        path.append(t)
    path.reverse()
    return [p + 1 for p in path], float(dist[d])


def k_shortest_paths(cost: np.ndarray, source: int, dest: int, k_paths: int) -> Tuple[List[List[int]], List[float]]:
    if source > cost.shape[0] or dest > cost.shape[0]:
        return [], []

    first_path, first_cost = dijkstra(cost, source, dest)
    if not first_path:
        return [], []

    paths: Dict[int, Tuple[List[int], float]] = {1: (first_path, first_cost)}
    current = 1
    candidates: List[Tuple[int, List[int], float]] = [(1, first_path, first_cost)]
    dev_source: Dict[int, int] = {1: first_path[0]}

    shortest_paths: List[List[int]] = [first_path]
    total_costs: List[float] = [first_cost]
    path_number = 1

    while len(shortest_paths) < k_paths and candidates:
        candidates = [c for c in candidates if c[0] != current]
        p_cur, _ = paths[current]
        w = dev_source[current]
        w_idx = p_cur.index(w)

        for idx_dev in range(w_idx, len(p_cur) - 1):
            temp = cost.copy()
            for i in range(idx_dev):
                v = p_cur[i] - 1
                temp[v, :] = INF
                temp[:, v] = INF

            same_sub = [p_cur]
            for sp in shortest_paths:
                if len(sp) >= idx_dev + 1 and sp[: idx_dev + 1] == p_cur[: idx_dev + 1]:
                    same_sub.append(sp)

            v_dev = p_cur[idx_dev] - 1
            for sp in same_sub:
                nxt = sp[idx_dev + 1] - 1
                temp[v_dev, nxt] = INF

            sub = p_cur[: idx_dev + 1]
            sub_cost = 0.0
            for i in range(len(sub) - 1):
                sub_cost += cost[sub[i] - 1, sub[i + 1] - 1]

            dev_path, dev_cost = dijkstra(temp, p_cur[idx_dev], dest)
            if dev_path:
                path_number += 1
                new_path = sub[:-1] + dev_path
                new_cost = sub_cost + dev_cost
                paths[path_number] = (new_path, new_cost)
                dev_source[path_number] = p_cur[idx_dev]
                candidates.append((path_number, new_path, new_cost))

        if not candidates:
            break

        best = min(candidates, key=lambda x: x[2])
        current = best[0]
        shortest_paths.append(paths[current][0])
        total_costs.append(paths[current][1])

    return shortest_paths, total_costs


def bpr(x: float, length: float, ffspeed: float, capacity: float) -> float:
    return (length / ffspeed) * (1.0 + 0.15 * (x / capacity) ** 4)


def initialize(data_dir: Path, re_assign: int = 0) -> Dict[str, object]:
    nw: Dict[str, object] = {}
    nw["BPR_piece_l"] = 1
    nw["BPR_piece_r"] = 4
    nw["BPR_piece"] = nw["BPR_piece_l"] + nw["BPR_piece_r"]
    nw["pce"] = np.array([1.0, 2.5])
    nw["xi"] = np.array([9.0, 38.0])

    test_id = np.arange(10, dtype=int)
    k_paths = 3
    no_class = 2
    no_od_on = len(test_id)
    path_vector = np.full(no_od_on, k_paths, dtype=int)

    network = read_table(data_dir / "Tilburg_network.txt")
    demand = read_table(data_dir / "Tilburg_demand.txt")
    node = read_table(data_dir / "Tilburg_node.txt")

    nlink = len(network["fromNode"])
    network["spd_range_c1"] = np.full(nlink, 0.4)
    network["spd_range_c2"] = np.full(nlink, 1.1)

    network["id"] = np.arange(1, nlink + 1, dtype=int)

    demand["demand_c"] = demand["demand_c"] * 1.0
    demand["demand_t"] = demand["demand_t"] * 1.0

    link_list = np.column_stack(
        [
            network["fromNode"],
            network["toNode"],
            network["length"],
            network["spdlimit_c"],
            network["spdlimit_t"],
            network["capacity"],
        ]
    )

    no_link = link_list.shape[0]
    no_node = int(np.max(link_list[:, 0]))
    od_list = np.column_stack(
        [
            demand["fromNode"][test_id],
            demand["toNode"][test_id],
            demand["demand_c"][test_id],
            demand["demand_t"][test_id],
        ]
    )

    source = demand["fromNode"][test_id].astype(int)
    destination = demand["toNode"][test_id].astype(int)

    def build_ff_cost(speed_col: str) -> np.ndarray:
        ff = np.full((no_node, no_node), INF)
        for m in range(no_link):
            i = int(network["fromNode"][m]) - 1
            j = int(network["toNode"][m]) - 1
            ff[i, j] = network["length"][m] / network[speed_col][m]
        return ff

    link_costff_c = build_ff_cost("spdlimit_c")
    link_costff_t = build_ff_cost("spdlimit_t")

    if re_assign != 0:
        raise RuntimeError("re_assign=1 path is not supported in the Python port.")

    shortest_c: List[List[List[int]]] = []
    shortest_t: List[List[List[int]]] = []
    for w in range(no_od_on):
        sp_c, _ = k_shortest_paths(link_costff_c, int(source[w]), int(destination[w]), int(path_vector[w]))
        sp_t, _ = k_shortest_paths(link_costff_t, int(source[w]), int(destination[w]), int(path_vector[w]))
        shortest_c.append(sp_c)
        shortest_t.append(sp_t)

    no_path = int(np.sum(path_vector))
    path_list = np.zeros((no_path, 2 + no_link, no_class), dtype=float)

    row = 0
    for w in range(no_od_on):
        for _ in range(path_vector[w]):
            path_list[row, 0, 0] = demand["fromNode"][test_id[w]]
            path_list[row, 1, 0] = demand["toNode"][test_id[w]]
            path_list[row, 0, 1] = demand["fromNode"][test_id[w]]
            path_list[row, 1, 1] = demand["toNode"][test_id[w]]
            row += 1

    def fill_path_incidence(paths_by_od: List[List[List[int]]], cls: int) -> None:
        start = 0
        for w in range(no_od_on):
            for p in range(path_vector[w]):
                path_nodes = paths_by_od[w][p]
                for r in range(len(path_nodes) - 1):
                    u = path_nodes[r]
                    v = path_nodes[r + 1]
                    for l in range(no_link):
                        if int(network["fromNode"][l]) == u and int(network["toNode"][l]) == v:
                            path_list[start + p, 2 + int(network["id"][l]) - 1, cls] = 1.0
                            break
            start += path_vector[w]

    fill_path_incidence(shortest_c, 0)
    fill_path_incidence(shortest_t, 1)

    nw.update(
        {
            "network": network,
            "demand": demand,
            "node": node,
            "link_list": link_list,
            "no_link": no_link,
            "no_node": no_node,
            "OD_list": od_list,
            "no_class": no_class,
            "no_OD": no_od_on,
            "path_vector": path_vector,
            "no_path": no_path,
            "path_list": path_list,
            "k_paths": k_paths,
            "test_id": test_id,
            "no_od_on": no_od_on,
        }
    )
    return nw


def build_and_solve(nw: Dict[str, object], timelimit: float = 2000.0) -> Tuple[np.ndarray, Dict[str, object]]:
    no_class = int(nw["no_class"])
    no_path = int(nw["no_path"])
    no_od = int(nw["no_OD"])
    no_link = int(nw["no_link"])
    bpr_piece = int(nw["BPR_piece"])

    ndv = no_class * no_path
    nadv = ndv + no_class * no_od + no_class * no_link + no_class * no_link * (bpr_piece + 1)
    nvar = ndv + nadv
    till_cstar = 2 * ndv
    till_clink = 2 * ndv + no_class * no_od
    till_sosind = nvar - no_class * no_link * (bpr_piece + 1)

    M = 300000.0
    Mc = 500000.0

    model = gp.Model("stm_mip_py")

    xvars: List[gp.Var] = []
    for j in range(nvar):
        vtype = GRB.CONTINUOUS
        if ndv <= j < 2 * ndv:
            vtype = GRB.BINARY
        xvars.append(model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=vtype, name=f"v_{j}"))

    model.update()

    pce = nw["pce"]
    path_list = nw["path_list"]
    od_list = nw["OD_list"]

    od_to_paths: Dict[Tuple[int, int], List[int]] = {}
    for p in range(no_path):
        od_key = (int(path_list[p, 0, 0]), int(path_list[p, 1, 0]))
        od_to_paths.setdefault(od_key, []).append(p)

    for m in range(no_class):
        for w in range(no_od):
            expr = gp.LinExpr()
            od_key = (int(od_list[w, 0]), int(od_list[w, 1]))
            for p in od_to_paths.get(od_key, []):
                expr += xvars[m * no_path + p]
            model.addConstr(expr == float(od_list[w, 2 + m]), name=f"flow_{m}_{w}")

    for m in range(no_class):
        for p in range(no_path):
            x_idx = m * no_path + p
            a_idx = ndv + m * no_path + p
            model.addConstr(-M * xvars[x_idx] + xvars[a_idx] <= 0.0, name=f"adv1_{m}_{p}")
            model.addConstr(xvars[x_idx] - M * xvars[a_idx] <= 0.0, name=f"adv2_{m}_{p}")

    for m in range(no_class):
        for p in range(no_path):
            od_key = (int(path_list[p, 0, m]), int(path_list[p, 1, m]))
            w = next(i for i in range(no_od) if int(od_list[i, 0]) == od_key[0] and int(od_list[i, 1]) == od_key[1])
            cstar_idx = till_cstar + m * no_od + w
            a_idx = ndv + m * no_path + p

            expr1 = gp.LinExpr()
            expr1 += xvars[cstar_idx]
            expr2 = gp.LinExpr()
            expr2 += Mc * xvars[a_idx] - xvars[cstar_idx]
            for e in range(no_link):
                inc = float(path_list[p, 2 + e, m])
                clink_idx = till_clink + m * no_link + e
                expr1 += -inc * xvars[clink_idx]
                expr2 += inc * xvars[clink_idx]
            model.addConstr(expr1 <= 0.0, name=f"ue1_{m}_{p}")
            model.addConstr(expr2 <= Mc, name=f"ue2_{m}_{p}")

    network = nw["network"]

    for m in range(no_class):
        for e in range(no_link):
            expr = gp.LinExpr()
            vars_sos: List[gp.Var] = []
            wts: List[float] = []
            step_capa = float(nw["link_list"][e, 5]) / float(nw["BPR_piece_l"])
            for l in range(bpr_piece + 1):
                idx = till_sosind + m * no_link * (bpr_piece + 1) + e * (bpr_piece + 1) + l
                expr += xvars[idx]
                vars_sos.append(xvars[idx])
                wts.append(step_capa * l)
            model.addConstr(expr == 1.0, name=f"sos_sum_b_{m}_{e}")
            model.addSOS(GRB.SOS_TYPE2, vars_sos, wts)

    for m in range(no_class):
        for e in range(no_link):
            expr = gp.LinExpr()
            step_capa = float(nw["link_list"][e, 5]) / float(nw["BPR_piece_l"])
            for l in range(bpr_piece + 1):
                idx = till_sosind + m * no_link * (bpr_piece + 1) + e * (bpr_piece + 1) + l
                expr += (step_capa * l) * xvars[idx]
            for m2 in range(no_class):
                for p in range(no_path):
                    inc = float(path_list[p, 2 + e, m2])
                    expr += -inc * float(pce[m2]) * xvars[m2 * no_path + p]
            model.addConstr(expr == 0.0, name=f"sos_sum_x_{m}_{e}")

    for m in range(no_class):
        for e in range(no_link):
            expr1 = gp.LinExpr()
            expr2 = gp.LinExpr()
            step_capa = float(nw["link_list"][e, 5]) / float(nw["BPR_piece_l"])
            for l in range(bpr_piece + 1):
                idx = till_sosind + m * no_link * (bpr_piece + 1) + e * (bpr_piece + 1) + l
                val = bpr(
                    step_capa * l,
                    float(nw["link_list"][e, 2]),
                    float(nw["link_list"][e, 3 + m]),
                    float(network["capacity"][e]),
                )
                expr1 += -val * xvars[idx]
                expr2 += val * xvars[idx]

            clink_idx = till_clink + m * no_link + e
            if m == 0:
                rho_minus = float(network["spd_range_c1"][e]) * float(network["spdlimit_c"][e])
                rho_plus = float(network["spd_range_c2"][e]) * float(network["spdlimit_c"][e])
            else:
                rho_minus = float(network.get("spd_range_t1", np.ones(no_link))[e]) * float(network["spdlimit_t"][e])
                rho_plus = float(network.get("spd_range_t2", np.ones(no_link))[e]) * float(network["spdlimit_t"][e])

            expr1 += rho_minus * xvars[clink_idx]
            expr2 += -rho_plus * xvars[clink_idx]
            model.addConstr(expr1 <= 0.0, name=f"sos_sum_c1_{m}_{e}")
            model.addConstr(expr2 <= 0.0, name=f"sos_sum_c2_{m}_{e}")

    obj = gp.LinExpr()
    for m in range(no_class):
        for w in range(no_od):
            cstar_idx = till_cstar + m * no_od + w
            obj += float(nw["xi"][m]) * float(od_list[w, 2 + m]) * xvars[cstar_idx]

    model.setObjective(obj, GRB.MINIMIZE)
    model.Params.TimeLimit = timelimit
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Optimization failed with status {model.Status}")

    sol = np.array([v.X for v in xvars], dtype=float)

    ctx = {
        "ndv": ndv,
        "nadv": nadv,
        "nvar": nvar,
        "till_cstar": till_cstar,
        "till_clink": till_clink,
        "till_sosind": till_sosind,
        "obj": float(model.ObjVal) if model.SolCount > 0 else math.nan,
        "status": int(model.Status),
    }
    return sol, ctx


def post_process(nw: Dict[str, object], x: np.ndarray, ctx: Dict[str, object]) -> Dict[str, object]:
    no_link = int(nw["no_link"])
    no_class = int(nw["no_class"])
    no_path = int(nw["no_path"])
    no_od = int(nw["no_OD"])

    ndv = int(ctx["ndv"])
    till_clink = int(ctx["till_clink"])
    till_sosind = int(ctx["till_sosind"])

    path_list = nw["path_list"]

    x_link = np.zeros((no_link, no_class), dtype=float)
    for m in range(no_class):
        for p in range(no_path):
            f = x[m * no_path + p]
            for e in range(no_link):
                x_link[e, m] += path_list[p, 2 + e, m] * f

    results = np.zeros((no_path, 4 * no_class + 2), dtype=float)
    results[:, : 2 * no_class] = np.reshape(x[: 2 * ndv], (no_path, 2 * no_class), order="F")

    for p in range(no_path):
        for m in range(no_class):
            for e in range(no_link):
                results[p, 4 + m] += x[till_clink + m * no_link + e] * path_list[p, 2 + e, m]

    rho = np.zeros((no_link, no_class), dtype=float)
    for e in range(no_link):
        step_capa = float(nw["link_list"][e, 5]) / float(nw["BPR_piece_l"])
        for m in range(no_class):
            denom = x[till_clink + m * no_link + e]
            if abs(denom) < 1e-12:
                continue
            for l in range(int(nw["BPR_piece"]) + 1):
                idx = till_sosind + m * no_link * (int(nw["BPR_piece"]) + 1) + e * (int(nw["BPR_piece"]) + 1) + l
                rho[e, m] += (
                    bpr(
                        step_capa * l,
                        float(nw["link_list"][e, 2]),
                        float(nw["link_list"][e, 3 + m]),
                        float(nw["network"]["capacity"][e]),
                    )
                    * x[idx]
                    / denom
                )

    x_link_total = x_link[:, 0] * float(nw["pce"][0]) + x_link[:, 1] * float(nw["pce"][1])

    real_link_cost = np.zeros((no_link, no_class), dtype=float)
    for l in range(no_link):
        for m in range(no_class):
            if rho[l, m] <= 0:
                continue
            speed = rho[l, m]
            real_link_cost[l, m] = (
                float(nw["network"]["length"][l])
                / speed
                * (1.0 + 0.15 * (x_link_total[l] / float(nw["network"]["capacity"][l])) ** 4)
            )

    real_path_cost = np.zeros((no_path, no_class), dtype=float)
    for p in range(no_path):
        for l in range(no_link):
            for m in range(no_class):
                real_path_cost[p, m] += real_link_cost[l, m] * path_list[p, 2 + l, m]

    for m in range(no_class):
        results[:, 6 + m] = real_path_cost[:, m]

    vh = np.zeros(no_class, dtype=float)
    for m in range(no_class):
        vh[m] = np.sum(results[:, m] * results[:, m + 6])
    vot = np.array([9.0, 38.0])
    vh_value = vh * vot

    return {
        "x_link": x_link,
        "results": results,
        "rho": rho,
        "real_link_cost": real_link_cost,
        "real_path_cost": real_path_cost,
        "vh": vh,
        "vh_value": vh_value,
        "vh_value_total": float(np.sum(vh_value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Python solver for multi-class Tilburg UE/BRUE models")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-agap", action="store_true")
    args = parser.parse_args()

    data_dir = args.project_root / "data" / "tilburg"
    nw = initialize(data_dir)
    x, ctx = build_and_solve(nw, timelimit=args.time_limit)
    post = post_process(nw, x, ctx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_agap:
        agap = calc_agap(
            nw=nw,
            results=post["results"],
            real_path_cost=post["real_path_cost"],
            real_link_cost=post["real_link_cost"],
            k_shortest_paths_func=k_shortest_paths,
        )
    else:
        agap = {"agap": math.nan, "agap_ori": math.nan, "agap_real": math.nan}

    if not args.skip_plots:
        plot_network(
            node=nw["node"],
            network=nw["network"],
            title="Tilburg network",
            out_file=args.output_dir / "network.png",
            show_label=True,
        )
        plot_loaded_links(
            node=nw["node"],
            network=nw["network"],
            load=np.round(post["x_link"][:, 0]),
            title="UE car flow (STM)",
            out_file=args.output_dir / "ue_car_flow.png",
            show_labels=True,
        )
        plot_loaded_links(
            node=nw["node"],
            network=nw["network"],
            load=np.round(post["x_link"][:, 1]),
            title="UE truck flow (STM)",
            out_file=args.output_dir / "ue_truck_flow.png",
            show_labels=True,
        )
        delta = np.round((post["rho"][:, 0] / nw["network"]["spdlimit_c"] - 1.0) * 100.0)
        plot_links_speed_limit_change(
            node=nw["node"],
            network=nw["network"],
            delta_pct=delta,
            title="Optimal Speed Limit Change in %",
            out_file=args.output_dir / "speed_limit_change.png",
            show_labels=True,
        )
        plot_computation_time(args.output_dir / "computation_time_log10.png")
        plot_link_capacity(
            node=nw["node"],
            network=nw["network"],
            out_file=args.output_dir / "link_capacity.png",
        )

    summary = {
        "status": int(ctx["status"]),
        "objective": float(ctx["obj"]),
        "vh_value_total": float(post["vh_value_total"]),
        "car_flow_total": float(np.sum(post["x_link"][:, 0])),
        "truck_flow_total": float(np.sum(post["x_link"][:, 1])),
        "agap": float(agap["agap"]),
        "agap_ori": float(agap["agap_ori"]),
        "agap_real": float(agap["agap_real"]),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"status={summary['status']}")
    print(f"objective={summary['objective']:.6f}")
    print(f"vh_value_total={summary['vh_value_total']:.6f}")
    print(f"car_flow_total={summary['car_flow_total']:.6f}")
    print(f"truck_flow_total={summary['truck_flow_total']:.6f}")
    print(f"agap={summary['agap']:.12f}")
    print(f"agap_ori={summary['agap_ori']:.12f}")
    print(f"agap_real={summary['agap_real']:.12f}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
