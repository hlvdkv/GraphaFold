#!/usr/bin/env python3
"""Compare the fixed strict GraphaFold run with per-PDB supplement metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRICS = ("F1", "INF", "PPV", "TPR")
S2_METHODS = ("GraphaFold_supplement", "SincFold", "SPOT_RNA", "UFold")
EXCLUDED = {"3OWZ", "6POM", "7QR3", "7QR4", "8BTZ", "8S95", "8UO6"}


def parse_s2(text: str) -> dict[str, dict[str, dict[str, float]]]:
    start = text.index(r"\label{S2_Table}")
    end = text.index(r"\end{longtable}", start)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for raw_line in text[start:end].splitlines():
        line = raw_line.strip()
        if not re.match(r"^[0-9A-Z]{4}\s*&", line):
            continue
        cells = [cell.strip() for cell in line.removesuffix(r"\\").split("&")]
        if len(cells) != 17:
            continue
        values = [float(value) for value in cells[1:]]
        result[cells[0]] = {
            method: dict(zip(METRICS, values[index * 4 : index * 4 + 4]))
            for index, method in enumerate(S2_METHODS)
        }
    return result


def read_fix_jsonl(path: Path) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    per_pdb: dict[str, dict[str, float]] = {}
    raw_rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pdb_id = str(row["sample_id"])
        if pdb_id in EXCLUDED:
            continue
        ppv = float(row["precision"])
        tpr = float(row["pipeline_recall"])
        per_pdb[pdb_id] = {
            "F1": float(row["pipeline_f1"]),
            "INF": math.sqrt(ppv * tpr) if ppv and tpr else 0.0,
            "PPV": ppv,
            "TPR": tpr,
        }
        raw_rows.append(row)
    return per_pdb, raw_rows


def macro(method_rows: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: sum(row[metric] for row in method_rows.values()) / len(method_rows)
        for metric in METRICS
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_average_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Macro-averaged per-structure performance on the 23-structure common subset.}",
        r"\label{tab:smallmed_macro_23}",
        r"\begin{tabular}{lcccc}",
        r"\thickhline",
        r"Method & F1 & INF & PPV & TPR \\",
        r"\thickhline",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        values = [f"{float(row[metric]):.3f}" for metric in METRICS]
        lines.append(" & ".join([method, *values]) + r" \\")
    lines.extend(
        [
            r"\thickhline",
            r"\end{tabular}",
            r"\end{table}",
            "% Only GraphaFold (new) uses strict AU/CG/GU-excluded labels.",
            "% Supplement methods retain the historical label definition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--fix-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    supplement = parse_s2(args.supplement.read_text())
    new, raw_rows = read_fix_jsonl(args.fix_jsonl)
    selected = sorted(new)
    if len(selected) != 23:
        raise ValueError(f"Expected 23 selected structures, found {len(selected)}")
    if set(selected) & EXCLUDED:
        raise ValueError("An excluded structure survived filtering")
    if any(float(row["total_positives"]) == 0 for row in raw_rows):
        raise ValueError("Selected set contains a structure with no strict positives")
    if missing := sorted(set(selected) - set(supplement)):
        raise ValueError(f"PDBs missing from supplement Table S2: {missing}")

    per_pdb_rows: list[dict[str, object]] = []
    for pdb_id in selected:
        row: dict[str, object] = {"PDB_ID": pdb_id}
        for metric in METRICS:
            row[f"GraphaFold_new_strict_{metric}"] = new[pdb_id][metric]
        for method in S2_METHODS:
            for metric in METRICS:
                row[f"{method}_legacy_{metric}"] = supplement[pdb_id][method][metric]
        per_pdb_rows.append(row)

    new_macro = macro(new)
    new_average: dict[str, object] = {
        "method": "GraphaFold (new)", "label_definition": "strict_AU_CG_GU_excluded", **new_macro
    }
    legacy_averages: list[dict[str, object]] = []
    for method in S2_METHODS:
        values = {pdb_id: supplement[pdb_id][method] for pdb_id in selected}
        legacy_averages.append(
            {
                "method": method,
                "label_definition": "legacy_AU_CG_GU_not_removed",
                **macro(values),
            }
        )
    # The requested table replaces the old GraphaFold row with the new model.
    averages: list[dict[str, object]] = [
        new_average,
        *(row for row in legacy_averages if row["method"] != "GraphaFold_supplement"),
    ]
    averages_with_old = [new_average, *legacy_averages]

    tp = sum(int(float(row["true_positives"])) for row in raw_rows)
    fp = sum(int(float(row["false_positives"])) for row in raw_rows)
    fn = sum(int(float(row["pipeline_false_negatives"])) for row in raw_rows)
    ppv = tp / (tp + fp)
    tpr = tp / (tp + fn)
    micro = [{
        "structures": len(raw_rows),
        "ground_truth": sum(int(float(row["total_positives"])) for row in raw_rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F1": 2 * ppv * tpr / (ppv + tpr),
        "INF": math.sqrt(ppv * tpr),
        "PPV": ppv,
        "TPR": tpr,
    }]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_pdb_fields = ["PDB_ID"]
    for method in ("GraphaFold_new_strict", *(f"{name}_legacy" for name in S2_METHODS)):
        per_pdb_fields.extend(f"{method}_{metric}" for metric in METRICS)
    write_csv(args.output_dir / "comparison_per_pdb_selected23.csv", per_pdb_rows, per_pdb_fields)
    write_csv(
        args.output_dir / "comparison_macro_average_selected23.csv",
        averages,
        ["method", "label_definition", *METRICS],
    )
    write_average_latex(args.output_dir / "comparison_macro_average_selected23.tex", averages)
    write_csv(
        args.output_dir / "comparison_macro_average_with_old_graphafold_selected23.csv",
        averages_with_old,
        ["method", "label_definition", *METRICS],
    )
    write_csv(
        args.output_dir / "graphafold_new_strict_micro_selected23.csv",
        micro,
        ["structures", "ground_truth", "TP", "FP", "FN", *METRICS],
    )

    readme = f"""# Fixed 23-structure comparison

Included PDBs: {', '.join(selected)}.

Excluded: {', '.join(sorted(EXCLUDED))}. `8BTZ` is excluded because it has no
positives under the strict non-canonical definition. Every included structure
has at least one strict positive.

The requested comparison is a macro-average: each PDB contributes equally and
F1/INF/PPV/TPR are averaged over the same 23 PDBs. New GraphaFold values come
from `{args.fix_jsonl}` and use strict labels. Values for the four supplement
methods come from rounded per-PDB Table S2 metrics and retain historical labels
that include some AU/CG/GU contacts. They are descriptive legacy baselines, not
a label-matched strict comparison. Raw prediction pairs are required to repair
those external per-PDB values.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(f"Wrote fixed comparison for {len(selected)} structures to {args.output_dir}")


if __name__ == "__main__":
    main()
