#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = [
    "UE",
    "UE_SO_E",
    "SC_UE_SO_E",
    "BRUE",
    "BRUE_SO",
    "BRUE_SO_E",
    "SC_BRUE_SO_E",
]


def fmt(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Build Tilburg-style comparison tables for Melbourne scenarios")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [data[s] for s in ORDER if s in data]
    if len(rows) < 7:
        raise RuntimeError("Expected at least the 7 core scenarios in summary JSON.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: travel time and emissions by class
    t1_csv = args.out_dir / "table_indicators_melbourne.csv"
    with t1_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Scenario",
            "TT Car (10^6 h)", "TT Truck (10^6 h)", "TT Total (10^6 h)",
            "Em Car (10^3 kg)", "Em Truck (10^3 kg)", "Em Total (10^3 kg)",
        ])
        for r in rows:
            w.writerow([
                r["scenario"],
                fmt(r["tt_car_h"] / 1e6), fmt(r["tt_truck_h"] / 1e6), fmt(r["tt_total_h"] / 1e6),
                fmt(r["em_car_ton"]), fmt(r["em_truck_ton"]), fmt(r["em_total_ton"]),
            ])

    # Table 2: monetary costs
    t2_csv = args.out_dir / "table_costs_melbourne.csv"
    with t2_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Scenario",
            "Time Cost Car (10^7 EUR)", "Time Cost Truck (10^7 EUR)", "Time Cost Total (10^7 EUR)",
            "Emission Cost Total (10^6 EUR)",
        ])
        for r in rows:
            w.writerow([
                r["scenario"],
                fmt(r["time_cost_car_eur"] / 1e7),
                fmt(r["time_cost_truck_eur"] / 1e7),
                fmt(r["time_cost_total_eur"] / 1e7),
                fmt(r["emission_cost_total_eur"] / 1e6),
            ])

    md = args.out_dir / "table_melbourne_overall.md"
    lines = []
    lines.append("## Melbourne Overall Comparison (Largest OD Slice)")
    lines.append("")
    lines.append("Travel times in 10^6 h, emissions in 10^3 kg (numerically equal to metric tons).")
    lines.append("")
    lines.append("| Scenario | TT Car | TT Truck | TT Total | Em Car | Em Truck | Em Total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {fmt(r['tt_car_h']/1e6)} | {fmt(r['tt_truck_h']/1e6)} | {fmt(r['tt_total_h']/1e6)} | {fmt(r['em_car_ton'])} | {fmt(r['em_truck_ton'])} | {fmt(r['em_total_ton'])} |"
        )

    lines.append("")
    lines.append("Time costs in 10^7 EUR, emission cost in 10^6 EUR.")
    lines.append("")
    lines.append("| Scenario | Time Cost Car | Time Cost Truck | Time Cost Total | Emission Cost Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {fmt(r['time_cost_car_eur']/1e7)} | {fmt(r['time_cost_truck_eur']/1e7)} | {fmt(r['time_cost_total_eur']/1e7)} | {fmt(r['emission_cost_total_eur']/1e6)} |"
        )

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote={t1_csv}")
    print(f"wrote={t2_csv}")
    print(f"wrote={md}")


if __name__ == "__main__":
    main()
