#!/usr/bin/env python3
"""
Analyze predictions.csv per-pcap and compute overall classification metrics.

Usage:
    python3 analyze_results.py <predictions_csv> <metadata_tsv> <output_dir> [run_metadata_json]

metadata_tsv columns (tab-separated, no header):
    pcap_name  true_label  start_row  end_row

run_metadata_json (optional): path to logs/run_metadata.json written by the
    controller; if omitted the script tries 'control_plane/logs/run_metadata.json'
    relative to the script's location.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)


# ── helpers ────────────────────────────────────────────────────────────────────

def separator(char="─", width=72):
    return char * width


def fmt_pct(n, total):
    if total == 0:
        return "0.00%"
    return f"{100 * n / total:.2f}%"


# ── main ───────────────────────────────────────────────────────────────────────

def _fmt_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def write_model_info(output_dir, run_metadata):
    """Write a human-readable model_info.txt into output_dir."""
    m = run_metadata
    lines = []
    sep = "═" * 72
    lines.append(sep)
    lines.append("  P4sec Pipeline — Model & Rules Metadata")
    lines.append(sep)

    # Reduction / transform
    method = m.get("reduction_method", "?").upper()
    n_comp = m.get("n_components", "?")
    bits   = m.get("pca_bits", "?")
    lines.append(f"\n  Reduction method : {method}")
    if method == "FEATURE_SELECTION":
        lines.append(f"  Selected features: {n_comp}")
    else:
        lines.append(f"  Components (k)   : {n_comp}  (quantised to {bits} bits)")

    # Support both old key names (pca_*) and new (transform_*)
    tr_per = m.get("transform_entries_per_component",
                   m.get("pca_entries_per_component", 0))
    tr_tot = m.get("transform_total_entries",
                   m.get("pca_total_entries", 0))
    if tr_per:
        lines.append(f"  Transform entries: {tr_per:,} per component × {n_comp} = {tr_tot:,} total")
    elif method == "FEATURE_SELECTION":
        lines.append(f"  Transform entries: none (feature selection — no transform tables)")
    lines.append(f"  Transform config size: {_fmt_bytes(m.get('transform_config_size_bytes', m.get('pca_config_size_bytes', 0)))}")

    # Classifier
    mtype = m.get("model_type", "?").upper()
    ddt   = m.get("declared_dt_type")
    if ddt and ddt != mtype.lower():
        mtype = f"{ddt.upper()} (via {mtype} proxy)"
    lines.append(f"\n  Classifier       : {mtype}")
    lines.append(f"  Model file       : {m.get('model_file', '?')}")
    lines.append(f"  Model file size  : {_fmt_bytes(m.get('model_size_bytes', 0))}")
    lines.append(f"  Model P4 entries : {m.get('model_entries', 0):,}")

    # Summary
    lines.append(f"\n  Total P4 entries : {m.get('total_table_entries', 0):,}")
    lines.append(f"  Total memory     : {_fmt_bytes(m.get('total_memory_bytes', 0))}")
    lines.append(f"  Rules load time  : {m.get('rules_load_time_s', 0):.2f} s")

    # Per-table breakdown
    tc = m.get("table_counts", {})
    if tc:
        lines.append("\n  Per-table entry counts:")
        for tbl, cnt in sorted(tc.items()):
            lines.append(f"    {tbl:<45s} {cnt:>8,}")

    lines.append("\n" + sep)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_info.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    predictions_csv  = sys.argv[1]
    metadata_tsv     = sys.argv[2]
    output_dir       = sys.argv[3]
    run_metadata_arg = sys.argv[4] if len(sys.argv) > 4 else None

    # ── Load run metadata (optional) ──────────────────────────────────────────
    run_metadata = None
    _candidates = []
    if run_metadata_arg:
        _candidates = [run_metadata_arg]
    else:
        # Try common relative locations
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _candidates = [
            os.path.join(_script_dir, 'control_plane', 'logs', 'run_metadata.json'),
            os.path.join(_script_dir, 'logs', 'run_metadata.json'),
        ]
    for _p in _candidates:
        if os.path.isfile(_p):
            try:
                with open(_p) as _f:
                    run_metadata = json.load(_f)
            except Exception as _e:
                print(f"WARNING: Could not parse run metadata {_p}: {_e}")
            break

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.isfile(predictions_csv):
        print(f"ERROR: predictions CSV not found: {predictions_csv}")
        sys.exit(1)

    df = pd.read_csv(predictions_csv)
    if "class_label" not in df.columns:
        print(f"ERROR: 'class_label' column not found in {predictions_csv}")
        sys.exit(1)

    metadata = []
    with open(metadata_tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                print(f"WARNING: skipping bad metadata line: {repr(line)}")
                continue
            pcap_name, true_label, start_row, end_row = parts
            metadata.append({
                "pcap":       pcap_name,
                "true_label": true_label,
                "start_row":  int(start_row),
                "end_row":    int(end_row),
            })

    if not metadata:
        print("ERROR: metadata file is empty or malformed.")
        sys.exit(1)

    # ── Per-file analysis ──────────────────────────────────────────────────────
    lines = []
    lines.append(separator("═"))
    lines.append("  P4sec Traffic Classification — Performance Report")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Predictions: {predictions_csv}")
    lines.append(separator("═"))

    all_true = []
    all_pred = []
    all_labels = set()

    for entry in metadata:
        pcap       = entry["pcap"]
        true_label = entry["true_label"]
        start      = entry["start_row"]
        end        = entry["end_row"]

        subset = df.iloc[start:end].copy()
        total  = len(subset)

        lines.append(f"\nFile : {pcap}")
        lines.append(f"  Ground-truth label : {true_label}")
        lines.append(f"  Flows classified   : {total}")

        if total == 0:
            lines.append("  (no flows captured for this file)")
            continue

        counts = subset["class_label"].value_counts().sort_index()
        all_labels.update(counts.index.tolist())
        all_labels.add(true_label)

        lines.append("  Label breakdown:")
        for label, cnt in counts.items():
            marker = " ✓" if label == true_label else "  "
            lines.append(f"    {marker}  {label:<20s} {cnt:>6d}  ({fmt_pct(cnt, total)})")

        correct  = counts.get(true_label, 0)
        accuracy = correct / total
        lines.append(f"  Accuracy : {accuracy:.4f}  ({fmt_pct(correct, total)})")
        lines.append(separator())

        all_true.extend([true_label] * total)
        all_pred.extend(subset["class_label"].tolist())

    # ── Overall metrics ────────────────────────────────────────────────────────
    lines.append("\n" + separator("═"))
    lines.append("  OVERALL METRICS")
    lines.append(separator("═"))

    if not all_true:
        lines.append("  No data — nothing to evaluate.")
    else:
        overall_acc = accuracy_score(all_true, all_pred)
        lines.append(f"  Overall accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
        lines.append(f"  Total flows      : {len(all_true)}")

        # Classification report (per-class precision / recall / F1)
        sorted_labels = sorted(all_labels)
        report = classification_report(
            all_true, all_pred,
            labels=sorted_labels,
            zero_division=0,
            digits=4,
        )
        lines.append("\n  Per-class metrics (precision / recall / F1 / support):")
        for rline in report.splitlines():
            lines.append("    " + rline)

        # Confusion matrix
        cm = confusion_matrix(all_true, all_pred, labels=sorted_labels)
        cm_df = pd.DataFrame(cm, index=sorted_labels, columns=sorted_labels)

        lines.append("\n  Confusion matrix  (rows = true label, cols = predicted label):")
        cm_str = cm_df.to_string()
        for cline in cm_str.splitlines():
            lines.append("    " + cline)

    lines.append("\n" + separator("═"))

    # ── Write outputs ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = os.path.join(output_dir, f"performance_{ts}.txt")
    report_text = "\n".join(lines) + "\n"

    with open(report_path, "w") as f:
        f.write(report_text)

    # Also save confusion matrix as CSV for easy import
    if all_true:
        cm_path = os.path.join(output_dir, f"confusion_matrix_{ts}.csv")
        cm_df.to_csv(cm_path)

        # Per-file summary CSV
        summary_rows = []
        for entry in metadata:
            pcap       = entry["pcap"]
            true_label = entry["true_label"]
            start      = entry["start_row"]
            end        = entry["end_row"]
            subset     = df.iloc[start:end]
            total      = len(subset)
            if total == 0:
                summary_rows.append({"pcap": pcap, "true_label": true_label,
                                     "total": 0, "correct": 0, "accuracy": 0.0})
                continue
            counts  = subset["class_label"].value_counts()
            correct = counts.get(true_label, 0)
            row = {"pcap": pcap, "true_label": true_label,
                   "total": total, "correct": correct,
                   "accuracy": correct / total}
            for lbl in sorted(all_labels):
                row[f"pred_{lbl}"] = int(counts.get(lbl, 0))
            summary_rows.append(row)

        summary_path = os.path.join(output_dir, f"per_file_summary_{ts}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # Write model info if metadata is available
    model_info_path = None
    if run_metadata:
        model_info_path = write_model_info(output_dir, run_metadata)

    # Print to terminal as well
    print(report_text)
    print(f"Report saved to      : {report_path}")
    if all_true:
        print(f"Confusion matrix CSV : {cm_path}")
        print(f"Per-file summary CSV : {summary_path}")
    if model_info_path:
        print(f"Model info           : {model_info_path}")


if __name__ == "__main__":
    main()
