#!/usr/bin/env python3
"""Redraw main-text (few traces) and supplementary (full set) figures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ANA = ROOT / "baseline_unbiased_md" / "analysis"
FIG = ANA / "figures"
MAIN = FIG / "main"
SI = FIG / "si"

COLORS = {
    "A_unbiased": "#3b6d99",
    "B_metad": "#c27a2d",
    "C_tmd": "#2e8b57",
    "D_ensemble": "#8b3a62",
    "E_stemper": "#6b4c9a",
    "F_gamd": "#b45309",
    "G_blind": "#0f766e",
    "H_wrong": "#9f1239",
    "I_bioemu": "#1d4ed8",
}
SHORT = {
    "A_unbiased": "A unbiased",
    "B_metad": "B WT-MetaD",
    "C_tmd": "C TMD",
    "D_ensemble": "D interpolation",
    "E_stemper": "E tempering",
    "F_gamd": "F GaMD",
    "G_blind": "G blind MetaD",
    "H_wrong": "H wrong-CV",
    "I_bioemu": "I BioEmu+MD",
}
MAIN_KEYS = ["A_unbiased", "B_metad", "C_tmd", "G_blind"]
RMSF_KEYS = ["A_unbiased", "B_metad", "C_tmd"]
PCA_KEYS = ["A_unbiased", "B_metad", "C_tmd"]
ALL_KEYS = list(SHORT)


def style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })


def finite_xy(t, y):
    xs, ys = [], []
    for a, b in zip(t, y):
        if b is None:
            continue
        xs.append(a)
        ys.append(b)
    return np.asarray(xs, float), np.asarray(ys, float)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    style()
    canvas = json.loads((ANA / "canvas_data.json").read_text())
    summary = json.loads((ANA / "summary.json").read_text())
    methods = canvas["methods"]
    t = np.asarray(canvas["time_cats"], float)
    res = np.asarray(canvas["rmsf_res"], int)

    # ----- Main Fig 2: coverage, 4 traces -----
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = [50, 100, 200, 300]
    for key in MAIN_KEYS:
        ys = [methods[key]["coverage_vs_ns"][str(c)] for c in xs]
        ax.plot(xs, ys, marker="o", color=COLORS[key], lw=1.8, label=SHORT[key])
    ax.set_xlabel("Cumulative production time (ns)")
    ax.set_ylabel("Occupied 0.5 Å bins")
    ax.set_title("Coverage of the NMP–CORE / LID–CORE plane")
    ax.legend(frameon=False)
    ax.set_xticks(xs)
    save(fig, MAIN / "coverage_main.png")

    # ----- SI coverage all 9 -----
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key in ALL_KEYS:
        ys = [methods[key]["coverage_vs_ns"][str(c)] for c in xs]
        ax.plot(xs, ys, marker="o", color=COLORS[key], lw=1.2, label=SHORT[key])
    ax.set_xlabel("Cumulative production time (ns)")
    ax.set_ylabel("Occupied 0.5 Å bins")
    ax.set_title("Coverage growth, all nine methods")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.set_xticks(xs)
    save(fig, SI / "coverage_all.png")

    # ----- Main Fig 3: RMSD to 1AKE only, 4 traces -----
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for key in MAIN_KEYS:
        tx, y = finite_xy(t, methods[key]["rmsd_closed_vs_t"])
        ax.plot(tx, y, color=COLORS[key], lw=1.7, label=SHORT[key])
    ax.axhline(3.5, color="0.45", ls="--", lw=1.0, label="closed cutoff 3.5 Å")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Cα RMSD to 1AKE (Å)")
    ax.set_title("Approach to the closed crystal")
    ax.set_ylim(0, 22)
    ax.legend(frameon=False, ncol=2)
    save(fig, MAIN / "rmsd_closed_main.png")

    # ----- SI: RMSD both crystals, all 9 -----
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    for key in ALL_KEYS:
        tx, yo = finite_xy(t, methods[key]["rmsd_open_vs_t"])
        _, yc = finite_xy(t, methods[key]["rmsd_closed_vs_t"])
        axes[0].plot(tx, yo, color=COLORS[key], lw=1.0, label=SHORT[key])
        axes[1].plot(tx, yc, color=COLORS[key], lw=1.0, label=SHORT[key])
    axes[1].axhline(3.5, color="0.45", ls="--", lw=0.9)
    axes[0].set_ylabel("Cα RMSD to 4AKE (Å)")
    axes[1].set_ylabel("Cα RMSD to 1AKE (Å)")
    axes[1].set_xlabel("Time (ns); D and I are concatenated seeds")
    axes[0].set_title("Global Cα RMSD, all nine methods")
    axes[0].legend(frameon=False, ncol=3, fontsize=7)
    save(fig, SI / "rmsd_time_all.png")

    # ----- SI: remaining methods RMSD to 1AKE -----
    rest = ["D_ensemble", "E_stemper", "F_gamd", "H_wrong", "I_bioemu"]
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for key in rest:
        tx, y = finite_xy(t, methods[key]["rmsd_closed_vs_t"])
        ax.plot(tx, y, color=COLORS[key], lw=1.5, label=SHORT[key])
    ax.axhline(3.5, color="0.45", ls="--", lw=1.0)
    ax.set_xlabel("Time (ns); D and I are concatenated seeds")
    ax.set_ylabel("Cα RMSD to 1AKE (Å)")
    ax.set_title("Methods omitted from main-text Figure 3")
    ax.set_ylim(0, 16)
    ax.legend(frameon=False, ncol=2)
    save(fig, SI / "rmsd_closed_rest.png")

    # ----- SI: RMSD to 4AKE for the four main methods -----
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for key in MAIN_KEYS:
        tx, y = finite_xy(t, methods[key]["rmsd_open_vs_t"])
        ax.plot(tx, y, color=COLORS[key], lw=1.6, label=SHORT[key])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Cα RMSD to 4AKE (Å)")
    ax.set_title("Distance from the open crystal (main-text methods)")
    ax.legend(frameon=False)
    save(fig, SI / "rmsd_open_main.png")

    # ----- Main Fig 6: RMSF A/B/C -----
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axvspan(30, 59, color="0.90", lw=0)
    ax.axvspan(122, 159, color="0.90", lw=0)
    ax.text(44.5, 0.16, "NMP", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(140.5, 0.16, "LID", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    for key in RMSF_KEYS:
        ax.plot(res, methods[key]["rmsf_A"], color=COLORS[key], lw=1.5, label=SHORT[key])
    ax.set_xlabel("Residue")
    ax.set_ylabel("Cα RMSF (Å)")
    ax.set_title("Flexibility after CORE superposition to 4AKE")
    ax.legend(frameon=False)
    save(fig, MAIN / "rmsf_main.png")

    # ----- SI RMSF all -----
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axvspan(30, 59, color="0.90", lw=0)
    ax.axvspan(122, 159, color="0.90", lw=0)
    ax.text(44.5, 0.16, "NMP", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(140.5, 0.16, "LID", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    for key in ALL_KEYS:
        ax.plot(res, methods[key]["rmsf_A"], color=COLORS[key], lw=1.05, label=SHORT[key])
    ax.set_xlabel("Residue")
    ax.set_ylabel("Cα RMSF (Å)")
    ax.set_title("Cα RMSF, all nine methods")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    save(fig, SI / "rmsf_all.png")

    # ----- Main PCA A/B/C -----
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    rng = np.random.default_rng(0)
    for key in PCA_KEYS:
        pc1 = np.asarray(methods[key]["pca_sample"]["pc1"])
        pc2 = np.asarray(methods[key]["pca_sample"]["pc2"])
        n = min(180, len(pc1))
        idx = np.sort(rng.choice(len(pc1), size=n, replace=False))
        ax.scatter(pc1[idx], pc2[idx], s=10, alpha=0.45, c=COLORS[key],
                   label=SHORT[key], linewidths=0)
    po, pc = canvas["pca_crystal"]["open"], canvas["pca_crystal"]["closed"]
    ax.scatter(po[0], po[1], c="white", edgecolors="black", s=70, zorder=4, label="4AKE")
    ax.scatter(pc[0], pc[1], c="red", edgecolors="black", s=70, zorder=4, label="1AKE")
    ax.set_xlabel(f"PC1 ({canvas['pca_var'][0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({canvas['pca_var'][1]:.1f}%)")
    ax.set_title("Joint Cα PCA (CORE-aligned)")
    ax.legend(frameon=False, markerscale=1.4, fontsize=8)
    save(fig, MAIN / "pca_main.png")

    # ----- SI PCA all -----
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for key in ALL_KEYS:
        pc1 = np.asarray(methods[key]["pca_sample"]["pc1"])
        pc2 = np.asarray(methods[key]["pca_sample"]["pc2"])
        n = min(120, len(pc1))
        idx = np.sort(rng.choice(len(pc1), size=n, replace=False))
        ax.scatter(pc1[idx], pc2[idx], s=7, alpha=0.35, c=COLORS[key],
                   label=SHORT[key], linewidths=0)
    ax.scatter(po[0], po[1], c="white", edgecolors="black", s=70, zorder=4, label="4AKE")
    ax.scatter(pc[0], pc[1], c="red", edgecolors="black", s=70, zorder=4, label="1AKE")
    ax.set_xlabel(f"PC1 ({canvas['pca_var'][0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({canvas['pca_var'][1]:.1f}%)")
    ax.set_title("Joint Cα PCA, all nine methods")
    ax.legend(frameon=False, markerscale=1.6, ncol=2, fontsize=7)
    save(fig, SI / "pca_all.png")

    # ----- SI BioEmu generated -----
    gen = summary["bioemu_generated"]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    labs = [f"{i:02d}" for i in range(gen["n_models"])]
    x = np.arange(len(labs))
    w = 0.36
    ax.bar(x - w / 2, gen["rmsd_closed_A"], width=w, label="to 1AKE", color="#2e8b57")
    ax.bar(x + w / 2, gen["rmsd_open_A"], width=w, label="to 4AKE", color="#3b6d99")
    ax.axhline(3.5, color="0.45", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_xlabel("BioEmu frame")
    ax.set_ylabel("Cα RMSD (Å)")
    ax.set_title("Generated models before MD relaxation")
    ax.legend(frameon=False)
    save(fig, SI / "bioemu_generated.png")

    # ----- SI stemper ladder -----
    st = summary["stemper_ladder"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    labs = [f"{t:.0f}" for t in st["temps_K"]]
    ax.bar(labs, [f * 100 for f in st["frac_by_state"]], color="#6b4c9a")
    ax.set_xlabel("Ladder temperature (K)")
    ax.set_ylabel("Occupancy (%)")
    ax.set_title("Simulated-tempering ladder occupancy")
    save(fig, SI / "stemper_ladder.png")

    # ----- Main Fig 4: RMSD-state bars with separators -----
    order = ALL_KEYS
    labs = [k[0] for k in order]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(order))
    bw = 0.24
    series = [
        ("open", [methods[k]["rmsd_frac"]["open"] * 100 for k in order], "#3b6d99"),
        ("closed", [methods[k]["rmsd_frac"]["closed"] * 100 for k in order], "#2e8b57"),
        ("intermediate", [methods[k]["rmsd_frac"]["intermediate"] * 100 for k in order], "#c27a2d"),
    ]
    for i, (name, ys, col) in enumerate(series):
        ax.bar(x + (i - 1) * bw, ys, width=bw, label=name, color=col, zorder=3)
    for i in range(len(order) - 1):
        ax.axvline(i + 0.5, color="0.55", ls="--", lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylabel("Frame fraction (%)")
    ax.set_xlabel("Method")
    ax.set_title("Open / closed / intermediate by Cα RMSD")
    ax.legend(frameon=False, loc="upper right")
    save(fig, MAIN / "rmsd_states.png")
    shutil.copy(MAIN / "rmsd_states.png", FIG / "rmsd_states.png")

    # ----- SI: CV-choice control (A / B / G / H) -----
    sm = summary["methods"]
    keys_cv = ["A_unbiased", "B_metad", "G_blind", "H_wrong"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    xlabs = ["A\nunbiased", "B\nfunctional", "G\nblind", "H\nwrong-CV"]
    xc = np.arange(len(keys_cv))
    axes[0].bar(xc, [sm[k]["rmsd_closed_min"] * 10 for k in keys_cv], color="#2e8b57")
    axes[0].axhline(3.5, color="0.45", ls="--", lw=1.0)
    axes[0].set_xticks(xc)
    axes[0].set_xticklabels(xlabs, fontsize=8)
    axes[0].set_ylabel("Min Cα RMSD to 1AKE (Å)")
    axes[0].set_title("Closest approach to closed")
    axes[1].bar(xc, [sm[k]["occupied_bins"] for k in keys_cv], color="#c27a2d")
    axes[1].set_xticks(xc)
    axes[1].set_xticklabels(xlabs, fontsize=8)
    axes[1].set_ylabel("Occupied CV bins")
    axes[1].set_title("Domain-plane coverage")
    save(fig, SI / "cv_controls.png")

    print("done")


if __name__ == "__main__":
    main()
