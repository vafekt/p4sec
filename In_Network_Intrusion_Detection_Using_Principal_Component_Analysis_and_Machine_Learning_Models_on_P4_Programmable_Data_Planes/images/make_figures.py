#!/usr/bin/env python3
"""Render publication-quality figures for the paper from
results/universal_summary.csv.

Conventions:
  - All percentage-style metrics are reported as values multiplied by 100
    with two decimals (e.g. 95.94, 0.985 -> 98.50).
  - No in-figure titles (captions only).
  - Em-dashes are not used; colons or commas separate clauses.
  - Tofino-specific budget annotations are *not* shown here (they belong
    to the future-work discussion in the article).
"""
import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE.parent.parent / "results" / "universal_summary.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

C_DT = "#0d47a1"
C_RF = "#c62828"
C_RAW_DT = "#1b5e20"
C_RAW_RF = "#ef6c00"
C_HIGHLIGHT = "#ffa000"


def load_rows():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                r[k] = None; continue
            try: r[k] = float(v)
            except: pass
    return rows


K_MAX = 15  # main sweep is k in 1..15; k=18 was exploratory only

def split(rows):
    pca_dt = sorted([r for r in rows if r["mode"] == "additive" and r["classifier"] == "dt"
                     and r["k"] is not None and int(r["k"]) <= K_MAX], key=lambda r: r["k"])
    pca_rf = sorted([r for r in rows if r["mode"] == "additive" and r["classifier"] == "rf"
                     and r["k"] is not None and int(r["k"]) <= K_MAX], key=lambda r: r["k"])
    raw_dt = next(r for r in rows if r["mode"] == "raw" and r["classifier"] == "dt")
    raw_rf = next(r for r in rows if r["mode"] == "raw" and r["classifier"] == "rf")
    return pca_dt, pca_rf, raw_dt, raw_rf


def save(fig, name):
    for ext in ("pdf", "png"):
        out = HERE / f"{name}.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


def style_xticks_all_k(ax):
    """Show every integer k from 1..15 on the x-axis."""
    ax.set_xticks(range(1, 16))
    ax.set_xlim(0.5, 15.5)


# ---------------------------------------------------------------------------
# Figure 1: in-distribution macro F1 vs k (single panel)
# ---------------------------------------------------------------------------
def fig_acc_vs_k(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    key = "cic_iot_macro_f1"
    ks_dt = [r["k"] for r in pca_dt]
    vs_dt = [r[key] * 100 for r in pca_dt]
    ks_rf = [r["k"] for r in pca_rf]
    vs_rf = [r[key] * 100 for r in pca_rf]
    ax.plot(ks_dt, vs_dt, "-o", color=C_DT, label="PCA + DT", lw=1.8, ms=5)
    ax.plot(ks_rf, vs_rf, "-s", color=C_RF, label="PCA + RF (4 trees)", lw=1.8, ms=5)
    v_raw_dt = raw_dt[key] * 100
    v_raw_rf = raw_rf[key] * 100
    ax.axhline(v_raw_dt, color=C_RAW_DT, ls="--", lw=1.4, alpha=0.85,
               label=f"Raw DT ({v_raw_dt:.2f})")
    ax.axhline(v_raw_rf, color=C_RAW_RF, ls=":", lw=1.4, alpha=0.85,
               label=f"Raw RF ({v_raw_rf:.2f})")
    ax.set_xlabel("Number of PCA components $k$")
    ax.set_ylabel("In-distribution macro F1 (%)")
    style_xticks_all_k(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    save(fig, "fig_acc_vs_k")


# ---------------------------------------------------------------------------
# Figure 2: in-distribution per-class F1 heat-map (2 decimals)
# ---------------------------------------------------------------------------
def fig_perclass_indist(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    classes = ["Benign", "BruteForce", "DoS", "Reconnaissance"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.6))
    fig.subplots_adjust(hspace=0.40)
    for ax, fam_pca, fam_raw, fam_label in [
        (axes[0], pca_dt, raw_dt, "Decision Tree, PCA components $k$"),
        (axes[1], pca_rf, raw_rf, "Random Forest (4 trees), PCA components $k$"),
    ]:
        labels = ["Raw"] + [f"{int(r['k'])}" for r in fam_pca]
        data = [[fam_raw[f"cic_iot_{c}_f1"] for c in classes]] + \
               [[r[f"cic_iot_{c}_f1"] for c in classes] for r in fam_pca]
        arr = np.array(data, dtype=float) * 100.0
        im = ax.imshow(arr.T, aspect="auto", cmap="viridis", vmin=60, vmax=100, interpolation="nearest")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_xlabel(fam_label)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if not np.isfinite(v): continue
                ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 80 else "black", fontsize=6.5)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="Per-class F1 (%)")
    save(fig, "fig_perclass_indist")


# ---------------------------------------------------------------------------
# Figure 3: cross-dataset macro F1 vs k (single panel)
# ---------------------------------------------------------------------------
def fig_xds_overall(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    key = "mu_iot_macro_f1"
    ks_dt = [r["k"] for r in pca_dt]
    vs_dt = [r[key] * 100 for r in pca_dt]
    ks_rf = [r["k"] for r in pca_rf]
    vs_rf = [r[key] * 100 for r in pca_rf]
    ax.plot(ks_dt, vs_dt, "-o", color=C_DT, label="PCA + DT", lw=1.8, ms=5)
    ax.plot(ks_rf, vs_rf, "-s", color=C_RF, label="PCA + RF (4 trees)", lw=1.8, ms=5)
    ax.axhline(raw_dt[key] * 100, color=C_RAW_DT, ls="--", lw=1.4, alpha=0.85,
               label=f"Raw DT ({raw_dt[key]*100:.2f})")
    ax.axhline(raw_rf[key] * 100, color=C_RAW_RF, ls=":", lw=1.4, alpha=0.85,
               label=f"Raw RF ({raw_rf[key]*100:.2f})")
    ax.set_xlabel("Number of PCA components $k$")
    ax.set_ylabel("Cross-dataset macro F1 (%)")
    style_xticks_all_k(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    save(fig, "fig_xds_overall")


# ---------------------------------------------------------------------------
# Figure 4: P4 footprint, classifier key width + entries (no Tofino budget)
# ---------------------------------------------------------------------------
def fig_p4_footprint(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    ks = [r["k"] for r in pca_dt]
    keybits = [r["classifier_key_bits"] for r in pca_dt]
    entries_dt = [r["total_table_entries"] for r in pca_dt]
    entries_rf = [r["total_table_entries"] for r in pca_rf]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    fig.subplots_adjust(wspace=0.30)

    ax1.bar(ks, keybits, color=C_DT, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax1.axhline(raw_dt["classifier_key_bits"], color=C_RAW_DT, ls="--", lw=1.4,
                label=f"Raw DT key, {int(raw_dt['classifier_key_bits'])} bits")
    ax1.set_xlabel("PCA components $k$")
    ax1.set_ylabel("Classifier match-key width (bits)")
    style_xticks_all_k(ax1)
    ax1.set_ylim(0, 540)
    ax1.legend(loc="upper left", frameon=True)

    width = 0.4
    ax2.bar([k - width / 2 for k in ks], entries_dt, width=width,
            color=C_DT, alpha=0.85, edgecolor="white", linewidth=0.6, label="PCA + DT")
    ax2.bar([k + width / 2 for k in ks], entries_rf, width=width,
            color=C_RF, alpha=0.85, edgecolor="white", linewidth=0.6, label="PCA + RF (4 trees)")
    ax2.axhline(raw_dt["total_table_entries"], color=C_RAW_DT, ls="--", lw=1.4,
                label=f"Raw DT, {int(raw_dt['total_table_entries'])} entries")
    ax2.axhline(raw_rf["total_table_entries"], color=C_RAW_RF, ls=":", lw=1.4,
                label=f"Raw RF, {int(raw_rf['total_table_entries'])} entries")
    ax2.set_xlabel("PCA components $k$")
    ax2.set_ylabel("Total P4 table entries")
    style_xticks_all_k(ax2)
    ax2.set_ylim(0, max(max(entries_dt), max(entries_rf)) * 1.45)
    ax2.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)
    save(fig, "fig_p4_footprint")


# ---------------------------------------------------------------------------
# Figure 5: Pareto plot for operating-point selection
# ---------------------------------------------------------------------------
def fig_pareto(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)

    def pareto_mask(pts):
        n = len(pts)
        mask = [True] * n
        for i in range(n):
            for j in range(n):
                if i == j: continue
                xi, yi = pts[i]; xj, yj = pts[j]
                if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                    mask[i] = False; break
        return mask

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    bundles = [
        ("PCA + DT", pca_dt, C_DT, "o"),
        ("PCA + RF (4 trees)", pca_rf, C_RF, "s"),
    ]
    for label, fam, color, marker in bundles:
        xs = [r["cic_iot_macro_f1"] * 100 for r in fam]
        ys = [r["mu_iot_macro_f1"] * 100 for r in fam]
        mask = pareto_mask(list(zip(xs, ys)))
        # non-dominated points are filled, dominated points hollow
        for i, (x, y) in enumerate(zip(xs, ys)):
            if mask[i]:
                ax.scatter([x], [y], c=color, marker=marker, s=55,
                           edgecolors="white", linewidths=0.6, zorder=4)
            else:
                ax.scatter([x], [y], facecolors="none", edgecolors=color,
                           marker=marker, s=40, linewidths=1.1, zorder=2)
        # label the chosen operating point (PCA + DT, k=7) once
        for r, x, y in zip(fam, xs, ys):
            if r["k"] == 7 and label == "PCA + DT":
                ax.scatter([x], [y], s=190, facecolors="none",
                           edgecolors=C_HIGHLIGHT, linewidths=2.0, zorder=5)
                ax.annotate(f"PCA + DT, $k$=7\nrecommended",
                            (x, y), xytext=(-110, -28),
                            textcoords="offset points",
                            fontsize=9, color=C_DT,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor=C_DT,
                                      lw=0.6, alpha=0.9),
                            arrowprops=dict(arrowstyle="->", color=C_DT, lw=0.8))
        ax.plot([], [], marker=marker, color=color, linestyle="None",
                markersize=6, label=label)

    # raw baselines as crosses
    ax.scatter([raw_dt["cic_iot_macro_f1"] * 100], [raw_dt["mu_iot_macro_f1"] * 100],
               marker="x", c=C_RAW_DT, s=80, linewidths=2.0, zorder=4, label="Raw DT")
    ax.scatter([raw_rf["cic_iot_macro_f1"] * 100], [raw_rf["mu_iot_macro_f1"] * 100],
               marker="+", c=C_RAW_RF, s=110, linewidths=2.0, zorder=4, label="Raw RF")

    ax.set_xlabel("In-distribution macro F1 (%)")
    ax.set_ylabel("Cross-dataset macro F1 (%)")
    ax.set_xlim(70, 100)
    ax.set_ylim(10, 55)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)
    save(fig, "fig_pareto")


# ---------------------------------------------------------------------------
# Legacy (kept in code but not invoked) — memory / load / tcam / phv / score
# ---------------------------------------------------------------------------
def fig_memory_load(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    ks = [r["k"] for r in pca_dt]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    ax1.plot(ks, [r["memory_KB_total"] for r in pca_dt], "-o", color=C_DT, label="PCA + DT", lw=1.8, ms=4)
    ax1.plot(ks, [r["memory_KB_total"] for r in pca_rf], "-s", color=C_RF, label="PCA + RF (4 trees)", lw=1.8, ms=4)
    ax1.axhline(raw_dt["memory_KB_total"], color=C_RAW_DT, ls="--", lw=1.4,
                label=f"Raw DT, {raw_dt['memory_KB_total']:.2f} KB")
    ax1.axhline(raw_rf["memory_KB_total"], color=C_RAW_RF, ls=":", lw=1.4,
                label=f"Raw RF, {raw_rf['memory_KB_total']:.2f} KB")
    ax1.set_xlabel("PCA components $k$")
    ax1.set_ylabel("Estimated rule memory (KB)")
    style_xticks_all_k(ax1)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    ax2.plot(ks, [r["rules_load_time_s"] for r in pca_dt], "-o", color=C_DT, label="PCA + DT", lw=1.8, ms=4)
    ax2.plot(ks, [r["rules_load_time_s"] for r in pca_rf], "-s", color=C_RF, label="PCA + RF (4 trees)", lw=1.8, ms=4)
    ax2.axhline(raw_dt["rules_load_time_s"], color=C_RAW_DT, ls="--", lw=1.4,
                label=f"Raw DT, {raw_dt['rules_load_time_s']:.2f} s")
    ax2.axhline(raw_rf["rules_load_time_s"], color=C_RAW_RF, ls=":", lw=1.4,
                label=f"Raw RF, {raw_rf['rules_load_time_s']:.2f} s")
    ax2.set_xlabel("PCA components $k$")
    ax2.set_ylabel("Rule-loading time (s)")
    style_xticks_all_k(ax2)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)
    save(fig, "fig_memory_load")


# ---------------------------------------------------------------------------
# Figure 6: TCAM expansion + PHV pressure (additional P4 metrics)
# ---------------------------------------------------------------------------
def fig_tcam_phv(rows):
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    ks = [r["k"] for r in pca_dt]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    ax1.plot(ks, [r["tcam_expansion_estimate"] for r in pca_dt], "-o", color=C_DT, label="PCA + DT", lw=1.8, ms=4)
    ax1.plot(ks, [r["tcam_expansion_estimate"] for r in pca_rf], "-s", color=C_RF, label="PCA + RF (4 trees)", lw=1.8, ms=4)
    ax1.axhline(raw_dt["tcam_expansion_estimate"], color=C_RAW_DT, ls="--", lw=1.4,
                label=f"Raw DT, {int(raw_dt['tcam_expansion_estimate'])}")
    ax1.set_xlabel("PCA components $k$")
    ax1.set_ylabel("Estimated TCAM expansion")
    ax1.set_yscale("log")
    style_xticks_all_k(ax1)
    ax1.legend(loc="upper right", frameon=True, fontsize=8)

    ax2.plot(ks, [r["phv_bits_pipeline"] for r in pca_dt], "-o", color=C_DT, label="PCA + DT", lw=1.8, ms=4)
    ax2.plot(ks, [r["phv_bits_pipeline"] for r in pca_rf], "-s", color=C_RF, label="PCA + RF (4 trees)", lw=1.8, ms=4)
    ax2.axhline(raw_dt["phv_bits_pipeline"], color=C_RAW_DT, ls="--", lw=1.4,
                label=f"Raw DT, {int(raw_dt['phv_bits_pipeline'])} bits")
    ax2.set_xlabel("PCA components $k$")
    ax2.set_ylabel("PHV pipeline pressure (bits)")
    style_xticks_all_k(ax2)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)
    save(fig, "fig_tcam_phv")


# ---------------------------------------------------------------------------
# Figure 7: optimal-combo "arch" — composite goodness curve over k
# ---------------------------------------------------------------------------
def fig_optimal_combo(rows):
    """Show an arch-shaped goodness curve to expose the best PCA k.
    The composite score balances in-distribution accuracy, cross-dataset
    accuracy, and footprint cost on a 0..100 scale:
      score = 0.40 * in_dist + 0.40 * xds + 0.20 * (1 - normalised_log_entries)
    The peak of the resulting curve is the recommended deployment k.
    """
    pca_dt, pca_rf, raw_dt, raw_rf = split(rows)
    ks = [r["k"] for r in pca_dt]

    def make_curve(fam):
        in_dist = np.array([r["cic_iot_macro_f1"] * 100 for r in fam])
        xds = np.array([r["mu_iot_macro_f1"] * 100 for r in fam])
        ents = np.array([r["total_table_entries"] for r in fam], dtype=float)
        log_ents = np.log10(ents)
        cost = (log_ents - log_ents.min()) / (log_ents.max() - log_ents.min() + 1e-9) * 100
        score = 0.40 * in_dist + 0.40 * xds + 0.20 * (100 - cost)
        return in_dist, xds, score

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for fam, color, marker, label in [(pca_dt, C_DT, "o", "PCA + DT"),
                                      (pca_rf, C_RF, "s", "PCA + RF (4 trees)")]:
        _, _, score = make_curve(fam)
        ax.plot(ks, score, "-" + marker, color=color, lw=1.8, ms=5, label=label)
        best = int(np.argmax(score))
        ax.scatter([ks[best]], [score[best]], s=180, facecolors="none",
                   edgecolors=C_HIGHLIGHT, linewidths=2.0, zorder=5)
        ax.annotate(f"best $k$={int(ks[best])}\nscore {score[best]:.2f}",
                    (ks[best], score[best]),
                    xytext=(10, 8 if color == C_DT else -28),
                    textcoords="offset points", color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color, lw=0.6, alpha=0.85))

    ax.set_xlabel("Number of PCA components $k$")
    ax.set_ylabel("Composite deployability score (%)")
    style_xticks_all_k(ax)
    ax.set_ylim(35, 90)
    ax.legend(loc="lower right", frameon=True)
    save(fig, "fig_optimal_combo")


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} configs from {CSV}")
    fig_acc_vs_k(rows)
    fig_perclass_indist(rows)
    fig_xds_overall(rows)
    fig_p4_footprint(rows)
    fig_pareto(rows)


if __name__ == "__main__":
    main()
