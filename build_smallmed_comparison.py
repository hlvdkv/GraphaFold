#!/usr/bin/env python3
"""Build the SmallMed comparison tables without mixing label definitions silently.

The supplement reports per-PDB metrics under the historical definition of a
"non-canonical" pair.  It does not report per-PDB predictions stratified by
base composition, so AU/CG/GU cannot be subtracted from those metrics.  This
script therefore labels the external per-PDB columns as legacy and separately
derives the exact all-44 strict aggregate that *can* be recovered from Table S6.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = ("F1", "INF", "PPV", "TPR")
S2_METHODS = ("GraphaFold_old", "SincFold", "SPOT_RNA", "UFold")
S6_METHODS = ("GraphaFold_old", "SPOT_RNA", "SincFold", "UFold")
CANONICAL_COMPOSITIONS = {"AU", "CG", "GU"}
EXCLUDED_PDBS = {"3OWZ", "6POM", "7QR3", "7QR4", "8S95", "8UO6"}


def section(text: str, start: str, end: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[left:right]


def latex_rows(block: str, expected_numeric_fields: int) -> list[tuple[str, list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not re.match(r"^[0-9A-Z]{2,5}\s*&", line):
            continue
        cells = [cell.strip() for cell in line.removesuffix(r"\\").split("&")]
        try:
            values = [float(cell) for cell in cells[1:]]
        except ValueError:
            continue
        if len(values) == expected_numeric_fields:
            rows.append((cells[0], values))
    return rows


def read_s2(text: str) -> dict[str, dict[str, dict[str, float]]]:
    block = section(text, r"\label{S2_Table}", r"\end{longtable}")
    parsed: dict[str, dict[str, dict[str, float]]] = {}
    for pdb_id, values in latex_rows(block, 16):
        parsed[pdb_id] = {}
        for method_i, method in enumerate(S2_METHODS):
            start = method_i * 4
            parsed[pdb_id][method] = dict(zip(METRICS, values[start : start + 4]))
    return parsed


def read_s6(text: str) -> list[dict[str, object]]:
    block = section(text, r"\label{S6_Table}", r"\end{longtable}")
    rows: list[dict[str, object]] = []
    for pair_type, values in latex_rows(block, 13):
        gt = int(values[0])
        methods: dict[str, dict[str, int]] = {}
        for method_i, method in enumerate(S6_METHODS):
            start = 1 + method_i * 3
            tp, table_fn, table_fp = map(int, values[start : start + 3])
            # Table S6 has its FN and FP column contents transposed: GT=TP+the
            # displayed FP in every row.  Interpret using this invariant.
            if tp + table_fp != gt:
                raise ValueError(f"Table S6 invariant failed for {pair_type}/{method}")
            methods[method] = {"TP": tp, "FP": table_fn, "FN": table_fp}
        rows.append({"pair_type": pair_type, "ground_truth": gt, "methods": methods})
    return rows


def read_new_graphafold(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["PDB_ID"]] = {metric: float(row[metric]) for metric in METRICS}
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    methods = ("GraphaFold_strict", "SincFold_legacy", "SPOT_RNA_legacy", "UFold_legacy")
    names = ("GraphaFold (strict)", "sincFold (legacy)", "SPOT-RNA (legacy)", "UFold (legacy)")
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{c|cccc|cccc|cccc|cccc}",
        r"\thickhline",
        " & " + " & ".join(rf"\multicolumn{{4}}{{c|}}{{{name}}}" for name in names) + r" \\",
        r"PDB ID & F1 & INF & PPV & TPR & F1 & INF & PPV & TPR & F1 & INF & PPV & TPR & F1 & INF & PPV & TPR \\",
        r"\thickhline",
    ]
    for row in rows:
        values = [str(row["PDB_ID"])]
        for method in methods:
            values.extend(fmt(float(row[f"{method}_{metric}"])) for metric in METRICS)
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\thickhline",
            r"\end{longtable}",
            r"\end{landscape}",
            "% IMPORTANT: external-method columns retain the historical label definition.",
            "% They are not directly comparable with the strict GraphaFold column.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def strict_all44(rows_s6: list[dict[str, object]]) -> list[dict[str, object]]:
    strict_rows = [row for row in rows_s6 if row["pair_type"] not in CANONICAL_COMPOSITIONS]
    total_gt = sum(int(row["ground_truth"]) for row in strict_rows)
    output: list[dict[str, object]] = []
    for method in S6_METHODS:
        tp = sum(int(row["methods"][method]["TP"]) for row in strict_rows)  # type: ignore[index]
        fp = sum(int(row["methods"][method]["FP"]) for row in strict_rows)  # type: ignore[index]
        fn = sum(int(row["methods"][method]["FN"]) for row in strict_rows)  # type: ignore[index]
        ppv = tp / (tp + fp) if tp + fp else 0.0
        tpr = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * ppv * tpr / (ppv + tpr) if ppv + tpr else 0.0
        output.append(
            {
                "method": method,
                "scope": "all_44_supplement_structures",
                "strict_ground_truth": total_gt,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "F1": f1,
                "INF": math.sqrt(ppv * tpr),
                "PPV": ppv,
                "TPR": tpr,
            }
        )
    return output


def selected_graphafold_summary(path: Path, selected: set[str]) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["sample_id"] in selected]
    if {row["sample_id"] for row in rows} != selected:
        raise ValueError("Molecule-metrics IDs do not match the selected comparison set")

    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    ppv = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * ppv * tpr / (ppv + tpr) if ppv + tpr else 0.0
    micro = {
        "averaging": "micro",
        "molecules": len(rows),
        "ground_truth": sum(int(row["ground_truth_count"]) for row in rows),
        "candidate_ground_truth": sum(int(row["candidate_ground_truth"]) for row in rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F1": f1,
        "INF": math.sqrt(ppv * tpr),
        "PPV": ppv,
        "TPR": tpr,
    }

    def value_or_zero(row: dict[str, str], key: str) -> float:
        return float(row[key]) if row[key] else 0.0

    macro = {
        "averaging": "macro_zero_for_undefined",
        "molecules": len(rows),
        "ground_truth": sum(int(row["ground_truth_count"]) for row in rows),
        "candidate_ground_truth": sum(int(row["candidate_ground_truth"]) for row in rows),
        "TP": "",
        "FP": "",
        "FN": "",
        "F1": sum(value_or_zero(row, "pipeline_f1") for row in rows) / len(rows),
        "INF": sum(value_or_zero(row, "INF") for row in rows) / len(rows),
        "PPV": sum(value_or_zero(row, "PPV") for row in rows) / len(rows),
        "TPR": sum(value_or_zero(row, "TPR") for row in rows) / len(rows),
    }
    return [micro, macro]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--graphafold-csv", type=Path, required=True)
    parser.add_argument("--molecule-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text = args.supplement.read_text()
    old = read_s2(text)
    new = read_new_graphafold(args.graphafold_csv)
    selected = sorted(set(new) - EXCLUDED_PDBS)
    missing = sorted(set(selected) - set(old))
    if missing:
        raise ValueError(f"Selected PDBs absent from Table S2: {missing}")

    combined: list[dict[str, object]] = []
    for pdb_id in selected:
        row: dict[str, object] = {"PDB_ID": pdb_id, "external_label_status": "legacy_AU_CG_GU_not_removed"}
        for metric in METRICS:
            row[f"GraphaFold_strict_{metric}"] = new[pdb_id][metric]
        for method in ("SincFold", "SPOT_RNA", "UFold"):
            for metric in METRICS:
                row[f"{method}_legacy_{metric}"] = old[pdb_id][method][metric]
        combined.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["PDB_ID", "external_label_status"]
    for method in ("GraphaFold_strict", "SincFold_legacy", "SPOT_RNA_legacy", "UFold_legacy"):
        fields.extend(f"{method}_{metric}" for metric in METRICS)
    write_csv(args.output_dir / "comparison_24_mixed_definitions.csv", combined, fields)
    write_latex(args.output_dir / "comparison_24_mixed_definitions.tex", combined)

    aggregate = strict_all44(read_s6(text))
    aggregate_fields = ["method", "scope", "strict_ground_truth", "TP", "FP", "FN", *METRICS]
    write_csv(args.output_dir / "legacy_methods_strict_aggregate_all44.csv", aggregate, aggregate_fields)

    selected_summary = selected_graphafold_summary(args.molecule_metrics, set(selected))
    summary_fields = [
        "averaging", "molecules", "ground_truth", "candidate_ground_truth",
        "TP", "FP", "FN", *METRICS,
    ]
    write_csv(args.output_dir / "graphafold_strict_selected24_summary.csv", selected_summary, summary_fields)

    note = f"""# Status of the SmallMed comparison

Selected PDBs: {len(selected)} ({', '.join(selected)}).

Excluded exactly as requested: {', '.join(sorted(EXCLUDED_PDBS))}.

`comparison_24_mixed_definitions.*` reconstructs the requested table layout. The
new GraphaFold column uses the strict definition (AU/CG/GU excluded), whereas
the three external-method columns are copied from Supplement Table S2 and still
use its historical labels. **Those columns must not be presented as a fair
strict comparison.** Table S2 contains only rounded per-PDB metrics, so the
AU/CG/GU contributions cannot be removed after the fact.

`legacy_methods_strict_aggregate_all44.csv` is an exact strict recomputation
possible from Supplement Table S6 after dropping AU, CG and GU. It covers all
44 supplement structures, not the selected 24. Table S6's displayed FN and FP
values are transposed (for every row, `ground truth = TP + displayed FP`); the
CSV corrects that transposition.

`graphafold_strict_selected24_summary.csv` contains the exact micro and macro
summary for the new GraphaFold on the selected 24 structures.

To obtain a publication-ready strict 24-PDB comparison, provide one predicted
pair list/contact map per PDB for sincFold, SPOT-RNA and UFold. The same strict
evaluator can then recompute all four methods from identical pairs and labels.
"""
    (args.output_dir / "README.md").write_text(note)

    print(f"Wrote {len(combined)} selected-PDB rows to {args.output_dir}")


if __name__ == "__main__":
    main()
