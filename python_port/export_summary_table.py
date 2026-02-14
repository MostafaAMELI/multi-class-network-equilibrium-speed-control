#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_ORDER = [
    "UE",
    "UE_SO",
    "UE_SO_E",
    "BRUE",
    "BRUE_SO",
    "BRUE_SO_E",
    "SC_UE_SO_E",
    "SC_BRUE_SO_E",
]


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object.")
    return data


def ordered_rows(data: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    names = [name for name in DEFAULT_ORDER if name in data]
    names += [name for name in data.keys() if name not in names]
    return [data[name] for name in names]


def write_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    fields = [
        "scenario",
        "status",
        "objective",
        "J1_time_cost",
        "J2_emission_cost",
        "equilibrium",
        "speed_control",
        "include_time",
        "include_emission",
        "epsilon",
        "theta",
        "car_flow_total",
        "truck_flow_total",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_markdown(rows: List[Dict[str, float]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Scenario",
        "Status",
        "Objective",
        "J1 Time Cost",
        "J2 Emission Cost",
        "Eq",
        "SC",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("scenario", "")),
                    str(r.get("status", "")),
                    f"{float(r.get('objective', 0.0)):.3f}",
                    f"{float(r.get('J1_time_cost', 0.0)):.3f}",
                    f"{float(r.get('J2_emission_cost', 0.0)):.3f}",
                    str(r.get("equilibrium", "")),
                    "yes" if bool(r.get("speed_control", False)) else "no",
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scenario summary JSON to CSV and Markdown tables.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "summary_v2_all.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "summary_v2_all.csv",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "summary_v2_all.md",
    )
    args = parser.parse_args()

    data = load_summary(args.input)
    rows = ordered_rows(data)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)

    print(f"wrote_csv={args.csv}")
    print(f"wrote_md={args.md}")


if __name__ == "__main__":
    main()
