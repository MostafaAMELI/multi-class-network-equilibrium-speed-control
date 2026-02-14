#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


def read_node_link_counts(data_dir: Path) -> Tuple[int, int]:
    with (data_dir / "node.csv").open("r", encoding="utf-8-sig", newline="") as f:
        n_nodes = sum(1 for _ in csv.DictReader(f))
    with (data_dir / "link.csv").open("r", encoding="utf-8-sig", newline="") as f:
        n_links = sum(1 for _ in csv.DictReader(f))
    return n_nodes, n_links


def read_od_stats(od_file: Path) -> Dict[str, float]:
    with od_file.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        dest_cols = [i for i, h in enumerate(header) if i > 0 and h.strip().lower() != "total"]
        dest_ids = [header[i].strip() for i in dest_cols]

        total = 0.0
        positive_pairs = 0
        origins_with_demand = set()
        dests_with_demand = set()

        for row in r:
            if not row:
                continue
            o = row[0].strip()
            row_sum = 0.0
            for i, d in zip(dest_cols, dest_ids):
                if i >= len(row):
                    continue
                s = row[i].strip()
                if not s:
                    continue
                v = float(s)
                total += v
                row_sum += v
                if v > 0:
                    positive_pairs += 1
                    dests_with_demand.add(d)
            if row_sum > 0:
                origins_with_demand.add(o)

    return {
        "total_demand_trips": total,
        "positive_od_pairs": float(positive_pairs),
        "active_origins": float(len(origins_with_demand)),
        "active_destinations": float(len(dests_with_demand)),
        "matrix_zones": float(len(dest_cols)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Create Melbourne scenario-characteristics table")
    p.add_argument("--summary", type=Path, required=True, help="Scenario summary JSON (single or bundle)")
    p.add_argument("--scenario", type=str, default="UE")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "testcases" / "4541153")
    p.add_argument("--od-slice", type=str, default="800_815")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "melbourne")
    args = p.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    scenario = data[args.scenario] if args.scenario in data and isinstance(data[args.scenario], dict) else data

    n_nodes, n_links = read_node_link_counts(args.data_dir)
    od_file = args.data_dir / f"OD_matrix_{args.od_slice}.csv"
    od_stats = read_od_stats(od_file)

    selected_od = scenario.get("selected_od", [])
    selected_total = float(sum(float(x.get("demand_total", 0.0)) for x in selected_od))
    truck_share = float(scenario.get("truck_share", 0.15))
    selected_truck = selected_total * truck_share
    selected_car = selected_total - selected_truck

    rows = [
        ("Scenario", scenario.get("scenario", args.scenario)),
        ("Equilibrium", scenario.get("equilibrium", "")),
        ("Speed control", str(bool(scenario.get("speed_control", False)))),
        ("OD slice", args.od_slice),
        ("Theta (EUR/ton CO2)", f"{float(scenario.get('theta', 70.0)):.2f}"),
        ("Epsilon (BRUE)", f"{float(scenario.get('epsilon', 0.10)):.4f}"),
        ("Network nodes (full)", str(n_nodes)),
        ("Network links (full)", str(n_links)),
        ("Network links (reduced model)", str(int(scenario.get("no_link_reduced", 0)))),
        ("OD matrix zones", str(int(od_stats["matrix_zones"]))),
        ("OD pairs with positive demand (full matrix)", str(int(od_stats["positive_od_pairs"]))),
        ("Active origins (full matrix)", str(int(od_stats["active_origins"]))),
        ("Active destinations (full matrix)", str(int(od_stats["active_destinations"]))),
        ("Total demand in OD slice (trips)", f"{od_stats['total_demand_trips']:.0f}"),
        ("OD pairs used in optimization", str(len(selected_od))),
        ("k-shortest paths per used OD", str(int(scenario.get("k_paths", 0)))),
        ("Top ODs requested", str(int(scenario.get("top_n_od", 0)))),
        ("Total demand used in optimization (trips)", f"{selected_total:.1f}"),
        ("Cars demand used (trips)", f"{selected_car:.1f}"),
        ("Trucks demand used (trips)", f"{selected_truck:.1f}"),
        ("Car link-flow total (veh)", f"{float(scenario.get('car_flow_total', 0.0)):.3f}"),
        ("Truck link-flow total (veh)", f"{float(scenario.get('truck_flow_total', 0.0)):.3f}"),
        ("TT total (h)", f"{float(scenario.get('tt_total_h', 0.0)):.6f}"),
        ("Emissions total (ton)", f"{float(scenario.get('em_total_ton', 0.0)):.6f}"),
        ("Time cost total (EUR)", f"{float(scenario.get('time_cost_total_eur', 0.0)):.6f}"),
        ("Emission cost total (EUR)", f"{float(scenario.get('emission_cost_total_eur', 0.0)):.6f}"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / f"scenario_characteristics_{args.scenario.lower()}_{args.od_slice}.csv"
    md_out = args.out_dir / f"scenario_characteristics_{args.scenario.lower()}_{args.od_slice}.md"

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Characteristic", "Value"])
        w.writerows(rows)

    lines = [f"## Scenario Characteristics: {args.scenario} ({args.od_slice})", "", "| Characteristic | Value |", "|---|---:|"]
    lines.extend([f"| {k} | {v} |" for k, v in rows])
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote={csv_out}")
    print(f"wrote={md_out}")


if __name__ == "__main__":
    main()
