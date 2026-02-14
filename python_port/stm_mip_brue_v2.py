#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from stm_mip_run import initialize

SCENARIOS = {
    "UE": {"equilibrium": "ue", "speed_control": False, "include_time": False, "include_emission": False},
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


def speed_bounds(network: Dict[str, np.ndarray], no_link: int, m: int, e: int, speed_control: bool) -> Tuple[float, float]:
    spd_range_t1 = network["spd_range_t1"] if "spd_range_t1" in network else np.ones(no_link)
    spd_range_t2 = network["spd_range_t2"] if "spd_range_t2" in network else np.ones(no_link)

    if speed_control:
        if m == 0:
            v_minus = float(network["spd_range_c1"][e]) * float(network["spdlimit_c"][e])
            v_plus = float(network["spd_range_c2"][e]) * float(network["spdlimit_c"][e])
        else:
            v_minus = float(spd_range_t1[e]) * float(network["spdlimit_t"][e])
            v_plus = float(spd_range_t2[e]) * float(network["spdlimit_t"][e])
    else:
        if m == 0:
            v0 = float(network["spdlimit_c"][e])
        else:
            v0 = float(network["spdlimit_t"][e])
        # Numerically robust near-fixed speed bounds for non-SC scenarios.
        v_minus = 0.999 * v0
        v_plus = 1.001 * v0

    v_minus = max(v_minus, 1e-3)
    v_plus = max(v_plus, v_minus + 1e-4)
    return v_minus, v_plus


def build_model(
    nw: Dict[str, object],
    equilibrium: str,
    speed_control: bool,
    include_time: bool,
    include_emission: bool,
    epsilon: float,
    theta: float,
    timelimit: float,
) -> Tuple[gp.Model, Dict[str, object]]:
    no_class = int(nw["no_class"])
    no_path = int(nw["no_path"])
    no_od = int(nw["no_OD"])
    no_link = int(nw["no_link"])
    bpr_piece = int(nw["BPR_piece"])
    pce = np.asarray(nw["pce"], dtype=float)
    xi = np.asarray(nw["xi"], dtype=float)
    path_list = np.asarray(nw["path_list"], dtype=float)
    od_list = np.asarray(nw["OD_list"], dtype=float)
    network = nw["network"]

    M = 500000.0
    Mflow = 300000.0

    od_to_paths: Dict[Tuple[int, int], List[int]] = {}
    path_od_idx: Dict[Tuple[int, int], int] = {}
    for p in range(no_path):
        od_key = (int(path_list[p, 0, 0]), int(path_list[p, 1, 0]))
        od_to_paths.setdefault(od_key, []).append(p)
    for w in range(no_od):
        path_od_idx[(int(od_list[w, 0]), int(od_list[w, 1]))] = w

    model = gp.Model("stm_mip_brue_v2")
    model.Params.TimeLimit = timelimit
    model.Params.NonConvex = 2

    f = model.addVars(no_class, no_path, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    a = model.addVars(no_class, no_path, vtype=GRB.BINARY, name="a")
    cstar = model.addVars(no_class, no_od, lb=0.0, vtype=GRB.CONTINUOUS, name="cstar")
    x_link = model.addVars(no_class, no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="x_link")
    x_agg = model.addVars(no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="x_agg")
    c_link = model.addVars(no_class, no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="c_link")
    c_path = model.addVars(no_class, no_path, lb=0.0, vtype=GRB.CONTINUOUS, name="c_path")

    lamb = model.addVars(no_link, bpr_piece + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="bpr_lam")
    bpr_val = model.addVars(no_link, lb=0.0, vtype=GRB.CONTINUOUS, name="bpr_val")

    em_piece = 8
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

    for e in range(no_link):
        cap = float(network["capacity"][e])
        length = float(network["length"][e])
        x_bp = [cap * l / float(nw["BPR_piece_l"]) for l in range(bpr_piece + 1)]
        y_bp = [length * (1.0 + 0.15 * (xx / cap) ** 4) for xx in x_bp]

        model.addConstr(gp.quicksum(lamb[e, l] for l in range(bpr_piece + 1)) == 1.0, name=f"lam_sum_{e}")
        model.addConstr(gp.quicksum(x_bp[l] * lamb[e, l] for l in range(bpr_piece + 1)) == x_agg[e], name=f"lam_x_{e}")
        model.addConstr(gp.quicksum(y_bp[l] * lamb[e, l] for l in range(bpr_piece + 1)) == bpr_val[e], name=f"lam_y_{e}")
        model.addSOS(GRB.SOS_TYPE2, [lamb[e, l] for l in range(bpr_piece + 1)], x_bp)

    for m in range(no_class):
        for e in range(no_link):
            v_minus, v_plus = speed_bounds(network, no_link, m, e, speed_control)
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

            model.addConstr(M * (c_path[m, p] - cstar[m, w]) >= 1.0 - a[m, p], name=f"eq_imp_{m}_{p}")
            model.addConstr(-Mflow * f[m, p] + a[m, p] <= 0.0, name=f"alink1_{m}_{p}")
            model.addConstr(f[m, p] - Mflow * a[m, p] <= 0.0, name=f"alink2_{m}_{p}")

    for m in range(no_class):
        for e in range(no_link):
            v_minus, v_plus = speed_bounds(network, no_link, m, e, speed_control)
            length = float(network["length"][e])
            x_ratio_max = float(bpr_piece) / float(nw["BPR_piece_l"])
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

    obj = gp.QuadExpr()
    if include_time:
        for m in range(no_class):
            for p in range(no_path):
                obj.add(float(xi[m]) * f[m, p] * c_path[m, p])
    if include_emission:
        for m in range(no_class):
            for e in range(no_link):
                obj.add(float(theta) * ebar[m, e] * x_link[m, e])

    model.setObjective(obj, GRB.MINIMIZE)

    handles = {
        "f": f,
        "a": a,
        "cstar": cstar,
        "x_link": x_link,
        "x_agg": x_agg,
        "c_link": c_link,
        "c_path": c_path,
        "ebar": ebar,
    }
    return model, handles


def solve(
    project_root: Path,
    scenario_name: str,
    equilibrium: str,
    speed_control: bool,
    include_time: bool,
    include_emission: bool,
    epsilon: float,
    theta: float,
    timelimit: float,
    output: Path,
) -> Dict[str, float]:
    nw = initialize(project_root / "data" / "tilburg")
    model, h = build_model(
        nw=nw,
        equilibrium=equilibrium,
        speed_control=speed_control,
        include_time=include_time,
        include_emission=include_emission,
        epsilon=epsilon,
        theta=theta,
        timelimit=timelimit,
    )
    model.optimize()

    if model.SolCount == 0:
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

    summary = {
        "status": int(model.Status),
        "objective": float(model.ObjVal),
        "J1_time_cost": j1,
        "J2_emission_cost": j2,
        "scenario": scenario_name,
        "equilibrium": equilibrium,
        "speed_control": bool(speed_control),
        "include_time": bool(include_time),
        "include_emission": bool(include_emission),
        "epsilon": float(epsilon),
        "theta": float(theta),
        "car_flow_total": float(np.sum(xlink_val[0, :])),
        "truck_flow_total": float(np.sum(xlink_val[1, :])),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BRUE/UE speed-control model (v2) with paper scenarios")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="SC_BRUE_SO_E")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Relative BRUE indifference band")
    parser.add_argument("--theta", type=float, default=70.0, help="Euro per ton CO2")
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "summary_v2.json")
    args = parser.parse_args()

    if args.all_scenarios:
        bundle: Dict[str, Dict[str, float]] = {}
        for name, cfg in SCENARIOS.items():
            out_i = args.output.parent / f"summary_v2_{name.lower()}.json"
            s = solve(
                project_root=args.project_root,
                scenario_name=name,
                equilibrium=cfg["equilibrium"],
                speed_control=cfg["speed_control"],
                include_time=cfg["include_time"],
                include_emission=cfg["include_emission"],
                epsilon=args.epsilon,
                theta=args.theta,
                timelimit=args.time_limit,
                output=out_i,
            )
            bundle[name] = s
            print(f"{name}: status={s['status']} objective={s['objective']:.6f}")

        combined = args.output.parent / "summary_v2_all.json"
        with combined.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        print(f"saved={combined}")
        return

    cfg = SCENARIOS[args.scenario]
    s = solve(
        project_root=args.project_root,
        scenario_name=args.scenario,
        equilibrium=cfg["equilibrium"],
        speed_control=cfg["speed_control"],
        include_time=cfg["include_time"],
        include_emission=cfg["include_emission"],
        epsilon=args.epsilon,
        theta=args.theta,
        timelimit=args.time_limit,
        output=args.output,
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
