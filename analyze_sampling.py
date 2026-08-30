#!/usr/bin/env python3
"""Compute paper-plan metrics from existing apo AdK trajectories.

Reads protein-only XTC from methods A–I (unbiased, MetaD, TMD, interpolations,
simulated tempering, GaMD, blind/wrong CVs, BioEmu+MD) and writes:
  analysis/summary.json
  analysis/canvas_data.json
  analysis/method_metrics.csv
  analysis/figures/*.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
FIG = OUT / "figures"
PROTEIN_PDB = ROOT / "output" / "protein.pdb"
OPEN_PDB = ROOT / "inputs" / "4AKE_protein.pdb"
CLOSED_PDB = ROOT / "inputs" / "1AKE_protein.pdb"

NMP_RES = set(range(30, 60))
LID_RES = set(range(122, 160))
CORE_RES = set(list(range(1, 30)) + list(range(60, 122)) + list(range(160, 215)))

DT_NS = 0.1  # TRAJ_INTERVAL_PS = 100
KT_KJ = 0.008314462618 * 300.0  # kJ/mol
BIN = 0.05  # nm for 2D occupancy
CV_LO, CV_HI = 1.2, 4.6
N_BINS = int(round((CV_HI - CV_LO) / BIN))
OPEN_RMSD_CUT = 0.35  # nm
CLOSED_RMSD_CUT = 0.35
SAMPLE_N = 250
TIME_CATS = list(range(0, 301, 5))


def ca_indices(top: md.Topology):
    idx = {"all": [], "core": [], "nmp": [], "lid": [], "resSeq": []}
    for atom in top.atoms:
        if atom.name != "CA":
            continue
        r = int(atom.residue.resSeq)
        idx["all"].append(atom.index)
        idx["resSeq"].append(r)
        if r in CORE_RES:
            idx["core"].append(atom.index)
        elif r in NMP_RES:
            idx["nmp"].append(atom.index)
        elif r in LID_RES:
            idx["lid"].append(atom.index)
    for key in ("all", "core", "nmp", "lid"):
        idx[key] = np.asarray(idx[key], dtype=int)
    idx["resSeq"] = np.asarray(idx["resSeq"], dtype=int)
    return idx


def load_ca_pdb(path: Path, topology: md.Topology) -> md.Trajectory:
    traj = md.load_pdb(str(path))
    # Match by residue id + CA, in topology order.
    ref_map = {}
    for atom in traj.topology.atoms:
        if atom.name == "CA":
            ref_map[int(atom.residue.resSeq)] = atom.index
    order = []
    for atom in topology.atoms:
        if atom.name != "CA":
            continue
        r = int(atom.residue.resSeq)
        if r not in ref_map:
            raise RuntimeError(f"{path.name} missing CA of residue {r}")
        order.append(ref_map[r])
    return traj.atom_slice(order)


def com(xyz: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    return xyz[:, atoms, :].mean(axis=1)


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def occupied_bins(x: np.ndarray, y: np.ndarray) -> int:
    ix = np.clip(((x - CV_LO) / BIN).astype(int), 0, N_BINS - 1)
    iy = np.clip(((y - CV_LO) / BIN).astype(int), 0, N_BINS - 1)
    return int(len(np.unique(ix * N_BINS + iy)))


def coverage_vs_time(d_nmp: np.ndarray, d_lid: np.ndarray, t: np.ndarray, cuts):
    out = []
    for c in cuts:
        m = t <= c + 1e-9
        out.append(occupied_bins(d_nmp[m], d_lid[m]) if m.any() else 0)
    return out


def pca_fit(x: np.ndarray, n=2):
    mean = x.mean(axis=0)
    xc = x - mean
    # economy SVD
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max(len(x) - 1, 1)
    var_ratio = var / var.sum()
    scores = xc @ vt[:n].T
    return scores, var_ratio[:n], mean, vt[:n]


def downsample_xy(t, y, cats):
    if len(t) == 0:
        return [None] * len(cats)
    out = []
    for c in cats:
        j = int(np.argmin(np.abs(t - c)))
        if abs(t[j] - c) > 2.5:
            out.append(None)
        else:
            out.append(float(y[j]))
    return out


def last_speed(csv_path: Path) -> float | None:
    if not csv_path.is_file():
        return None
    speeds = []
    with csv_path.open() as handle:
        rows = list(csv.reader(handle))
    for row in rows[1:]:
        if len(row) < 8:
            continue
        try:
            v = float(row[7])
        except ValueError:
            continue
        if v > 1:
            speeds.append(v)
    return float(np.median(speeds[-20:])) if speeds else None


def elapsed_hours(csv_path: Path) -> float | None:
    if not csv_path.is_file():
        return None
    last = None
    with csv_path.open() as handle:
        rows = list(csv.reader(handle))
    for row in rows[1:]:
        if len(row) < 9:
            continue
        try:
            last = float(row[8])
        except ValueError:
            continue
    if last is None or last <= 0:
        return None
    return last / 3600.0


def mean_energy(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    e, temp = [], []
    with csv_path.open() as handle:
        rows = list(csv.reader(handle))
    for row in rows[1:]:
        if len(row) < 5:
            continue
        try:
            e.append(float(row[3]))
            temp.append(float(row[4]))
        except ValueError:
            continue
    if not e:
        return {}
    return {
        "E_pot_mean": float(np.mean(e)),
        "E_pot_std": float(np.std(e)),
        "T_mean": float(np.mean(temp)),
        "T_std": float(np.std(temp)),
    }


def load_method_traj(label: str, xtcs: list[Path], topology: md.Topology, idx):
    pieces = []
    t_off = 0.0
    seed_id = []
    for si, path in enumerate(xtcs):
        if not path.is_file():
            continue
        tr = md.load(str(path), top=str(PROTEIN_PDB), atom_indices=idx["all"])
        n = tr.n_frames
        t = t_off + np.arange(n, dtype=float) * DT_NS
        pieces.append((tr.xyz.copy(), t, np.full(n, si, dtype=int)))
        t_off = float(t[-1] + DT_NS) if n else t_off
        print(f"  {label} {path.parent.name}: {n} frames, {n * DT_NS:.1f} ns", flush=True)
    if not pieces:
        raise FileNotFoundError(label)
    xyz = np.concatenate([p[0] for p in pieces], axis=0)
    t = np.concatenate([p[1] for p in pieces], axis=0)
    seed = np.concatenate([p[2] for p in pieces], axis=0)
    return xyz, t, seed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    top_full = md.load_pdb(str(PROTEIN_PDB))
    idx = ca_indices(top_full.topology)
    print(
        f"CA atoms: all={len(idx['all'])} CORE={len(idx['core'])} "
        f"NMP={len(idx['nmp'])} LID={len(idx['lid'])}",
        flush=True,
    )
    # Map domain atoms in the CA-only array
    ca_res = idx["resSeq"]
    core_ca = np.array([i for i, r in enumerate(ca_res) if r in CORE_RES], dtype=int)
    nmp_ca = np.array([i for i, r in enumerate(ca_res) if r in NMP_RES], dtype=int)
    lid_ca = np.array([i for i, r in enumerate(ca_res) if r in LID_RES], dtype=int)

    open_ca = load_ca_pdb(OPEN_PDB, top_full.topology)
    closed_ca = load_ca_pdb(CLOSED_PDB, top_full.topology)
    dummy_top = open_ca.topology

    def metrics_from_xyz(xyz):
        tr = md.Trajectory(xyz=xyz, topology=dummy_top)
        d_nmp = dist(com(xyz, nmp_ca), com(xyz, core_ca))
        d_lid = dist(com(xyz, lid_ca), com(xyz, core_ca))
        rmsd_open = md.rmsd(tr, open_ca, 0)  # nm
        rmsd_closed = md.rmsd(tr, closed_ca, 0)
        # CORE-aligned coordinates for PCA / RMSF
        aligned = tr.superpose(open_ca, atom_indices=core_ca)
        return {
            "d_nmp": d_nmp.astype(np.float64),
            "d_lid": d_lid.astype(np.float64),
            "rmsd_open": rmsd_open.astype(np.float64),
            "rmsd_closed": rmsd_closed.astype(np.float64),
            "xyz_al": aligned.xyz.copy(),
        }

    crystal = {
        "open": metrics_from_xyz(open_ca.xyz),
        "closed": metrics_from_xyz(closed_ca.xyz),
    }
    # drop xyz from crystal dict for json later
    crystal_cv = {
        "open": {
            "d_nmp": float(crystal["open"]["d_nmp"][0]),
            "d_lid": float(crystal["open"]["d_lid"][0]),
            "rmsd_open": 0.0,
            "rmsd_closed": float(crystal["open"]["rmsd_closed"][0]),
        },
        "closed": {
            "d_nmp": float(crystal["closed"]["d_nmp"][0]),
            "d_lid": float(crystal["closed"]["d_lid"][0]),
            "rmsd_open": float(crystal["closed"]["rmsd_open"][0]),
            "rmsd_closed": 0.0,
        },
    }
    print("crystal CVs", json.dumps(crystal_cv, indent=2), flush=True)

    mid_nmp = 0.5 * (crystal_cv["open"]["d_nmp"] + crystal_cv["closed"]["d_nmp"])
    mid_lid = 0.5 * (crystal_cv["open"]["d_lid"] + crystal_cv["closed"]["d_lid"])

    methods = {
        "A_unbiased": {
            "title": "A unbiased MD",
            "xtcs": [ROOT / "output" / "prod_protein.xtc"],
            "thermo": [ROOT / "output" / "prod_thermo.csv"],
            "color": "#3b6d99",
        },
        "B_metad": {
            "title": "B WT-MetaD",
            "xtcs": [ROOT / "output_metad" / "prod_protein.xtc"],
            "thermo": [ROOT / "output_metad" / "prod_thermo.csv"],
            "color": "#c27a2d",
        },
        "C_tmd": {
            "title": "C targeted MD",
            "xtcs": [ROOT / "output_tmd" / "prod_protein.xtc"],
            "thermo": [ROOT / "output_tmd" / "prod_thermo.csv"],
            "color": "#2e8b57",
        },
        "D_ensemble": {
            "title": "D interpolation MD",
            "xtcs": [ROOT / "output_ai" / f"seed_{i:02d}" / "prod_protein.xtc" for i in range(6)],
            "thermo": [ROOT / "output_ai" / f"seed_{i:02d}" / "prod_thermo.csv" for i in range(6)],
            "color": "#8b3a62",
            "concat": True,
        },
        "E_stemper": {
            "title": "E simulated tempering",
            "xtcs": [ROOT / "output_stemper" / "prod_protein.xtc"],
            "thermo": [ROOT / "output_stemper" / "prod_thermo.csv"],
            "color": "#6b4c9a",
        },
        "F_gamd": {
            "title": "F dihedral GaMD",
            "xtcs": [ROOT / "output_gamd" / "prod_protein.xtc"],
            "thermo": [ROOT / "output_gamd" / "prod_thermo.csv"],
            "color": "#b45309",
        },
        "G_blind": {
            "title": "G blind MetaD",
            "xtcs": [ROOT / "output_blind" / "prod_protein.xtc"],
            "thermo": [ROOT / "output_blind" / "prod_thermo.csv"],
            "color": "#0f766e",
        },
        "H_wrong": {
            "title": "H wrong-CV MetaD",
            "xtcs": [ROOT / "output_wrong" / "prod_protein.xtc"],
            "thermo": [ROOT / "output_wrong" / "prod_thermo.csv"],
            "color": "#9f1239",
        },
        "I_bioemu": {
            "title": "I BioEmu + MD",
            "xtcs": [ROOT / "output_bioemu" / f"seed_{i:02d}" / "prod_protein.xtc" for i in range(6)],
            "thermo": [ROOT / "output_bioemu" / f"seed_{i:02d}" / "prod_thermo.csv" for i in range(6)],
            "color": "#1d4ed8",
            "concat": True,
        },
    }
    concat_keys = {k for k, s in methods.items() if s.get("concat")}

    data = {}
    for key, spec in methods.items():
        print(f"loading {key} ...", flush=True)
        xyz, t, seed = load_method_traj(key, spec["xtcs"], top_full.topology, idx)
        m = metrics_from_xyz(xyz)
        m["t"] = t
        m["seed"] = seed
        m["n_frames"] = int(len(t))
        m["sampled_ns"] = float(len(t) * DT_NS)
        data[key] = m

    # Combined PCA on CORE-aligned CA
    blocks = []
    owners = []
    for key, m in data.items():
        x = m["xyz_al"].reshape(len(m["t"]), -1)
        blocks.append(x)
        owners.append(np.full(len(x), key, dtype=object))
    x_all = np.vstack(blocks)
    # add crystals
    x_open = crystal["open"]["xyz_al"].reshape(1, -1)
    x_closed = crystal["closed"]["xyz_al"].reshape(1, -1)
    x_fit = np.vstack([x_all, x_open, x_closed])
    scores, var_ratio, pca_mean, pca_comp = pca_fit(x_fit, 2)
    n_all = len(x_all)
    pc_all = scores[:n_all]
    pc_open = scores[n_all]
    pc_closed = scores[n_all + 1]
    offset = 0
    for key, m in data.items():
        n = len(m["t"])
        m["pc1"] = pc_all[offset : offset + n, 0]
        m["pc2"] = pc_all[offset : offset + n, 1]
        offset += n

    # RMSF vs open CORE-aligned
    rmsf = {}
    for key, m in data.items():
        # nm
        rmsf[key] = m["xyz_al"].std(axis=0).mean(axis=1) * 10.0  # Angstrom

    def classify(m):
        nmp_closed = m["d_nmp"] < mid_nmp
        lid_closed = m["d_lid"] < mid_lid
        quad = np.empty(len(m["t"]), dtype=object)
        quad[(~lid_closed) & (~nmp_closed)] = "OO"
        quad[(~lid_closed) & nmp_closed] = "OC"  # LID open, NMP closed
        quad[lid_closed & (~nmp_closed)] = "CO"  # LID closed, NMP open
        quad[lid_closed & nmp_closed] = "CC"
        open_like = m["rmsd_open"] < OPEN_RMSD_CUT
        closed_like = m["rmsd_closed"] < CLOSED_RMSD_CUT
        both = open_like & closed_like
        rmsd_state = np.full(len(m["t"]), "intermediate", dtype=object)
        rmsd_state[open_like & ~closed_like] = "open"
        rmsd_state[closed_like & ~open_like] = "closed"
        rmsd_state[both & (m["rmsd_open"] <= m["rmsd_closed"])] = "open"
        rmsd_state[both & (m["rmsd_closed"] < m["rmsd_open"])] = "closed"
        return quad, rmsd_state

    def n_transitions(states):
        if len(states) < 2:
            return 0
        return int(np.sum(states[1:] != states[:-1]))

    summary = {
        "system": "apo E. coli AdK (4AKE chain A)",
        "note": (
            "TMD is nonequilibrium (biased toward 1AKE). WT-MetaD / blind / wrong-CV "
            "FES traces are biased occupancy, not reweighted ΔG. Method D uses "
            "open→closed interpolations. Method I is BioEmu generative models plus "
            "300 ns MD relaxation (6 × 50 ns)."
        ),
        "crystal": crystal_cv,
        "thresholds_nm": {
            "mid_nmp": float(mid_nmp),
            "mid_lid": float(mid_lid),
            "open_rmsd_cut": OPEN_RMSD_CUT,
            "closed_rmsd_cut": CLOSED_RMSD_CUT,
            "bin_nm": BIN,
        },
        "pca_var_ratio": [float(x) for x in var_ratio],
        "methods": {},
    }

    cuts = [50, 100, 200, 300]
    max_occ = 1
    method_rows = []
    for key, spec in methods.items():
        m = data[key]
        quad, rmsd_state = classify(m)
        m["quad"] = quad
        m["rmsd_state"] = rmsd_state
        occ = occupied_bins(m["d_nmp"], m["d_lid"])
        max_occ = max(max_occ, occ)
        speeds = [last_speed(p) for p in spec["thermo"] if p.is_file()]
        speeds = [s for s in speeds if s]
        hours = [elapsed_hours(p) for p in spec["thermo"] if p.is_file()]
        hours = [h for h in hours if h]
        energies = [mean_energy(p) for p in spec["thermo"] if p.is_file()]
        e_mean = float(np.mean([e["E_pot_mean"] for e in energies if e])) if energies else None
        t_mean = float(np.mean([e["T_mean"] for e in energies if e])) if energies else None
        ns_day = float(np.median(speeds)) if speeds else None
        gpu_h = float(m["sampled_ns"] / ns_day * 24.0) if ns_day else None
        wall_h = float(np.sum(hours)) if hours else gpu_h
        frac_quad = {s: float(np.mean(quad == s)) for s in ("OO", "OC", "CO", "CC")}
        frac_rmsd = {s: float(np.mean(rmsd_state == s)) for s in ("open", "closed", "intermediate")}
        t_for_cov = m["t"] if key not in concat_keys else np.arange(len(m["t"]), dtype=float) * DT_NS
        cov_t = coverage_vs_time(m["d_nmp"], m["d_lid"], t_for_cov, cuts)

        # concerted vs sequential: occupancy of mixed quadrants vs CC/OO
        mixed = frac_quad["OC"] + frac_quad["CO"]
        # Pearson of CVs
        if m["d_nmp"].std() > 1e-8 and m["d_lid"].std() > 1e-8:
            corr = float(np.corrcoef(m["d_nmp"], m["d_lid"])[0, 1])
        else:
            corr = 0.0

        # first hitting times (single trajectory methods)
        hit_closed_rmsd = None
        hit_open_rmsd = None
        if key not in concat_keys:
            ic = np.where(rmsd_state == "closed")[0]
            io = np.where(rmsd_state == "open")[0]
            if len(ic):
                hit_closed_rmsd = float(m["t"][ic[0]])
            if len(io):
                hit_open_rmsd = float(m["t"][io[0]])
        else:
            for s in np.unique(m["seed"]):
                mask = m["seed"] == s
                t_local = np.arange(int(mask.sum()), dtype=float) * DT_NS
                st = rmsd_state[mask]
                ic = np.where(st == "closed")[0]
                io = np.where(st == "open")[0]
                if len(ic):
                    hit = float(t_local[ic[0]])
                    hit_closed_rmsd = hit if hit_closed_rmsd is None else min(hit_closed_rmsd, hit)
                if len(io):
                    hit = float(t_local[io[0]])
                    hit_open_rmsd = hit if hit_open_rmsd is None else min(hit_open_rmsd, hit)

        rec = {
            "title": spec["title"],
            "n_frames": m["n_frames"],
            "sampled_ns": round(m["sampled_ns"], 1),
            "ns_per_day": None if ns_day is None else round(ns_day, 1),
            "gpu_hours_eq": None if gpu_h is None else round(gpu_h, 2),
            "wall_hours_log": None if wall_h is None else round(wall_h, 2),
            "E_pot_mean": None if e_mean is None else round(e_mean, 1),
            "T_mean": None if t_mean is None else round(t_mean, 2),
            "d_nmp_mean": round(float(m["d_nmp"].mean()), 3),
            "d_nmp_min": round(float(m["d_nmp"].min()), 3),
            "d_nmp_max": round(float(m["d_nmp"].max()), 3),
            "d_lid_mean": round(float(m["d_lid"].mean()), 3),
            "d_lid_min": round(float(m["d_lid"].min()), 3),
            "d_lid_max": round(float(m["d_lid"].max()), 3),
            "rmsd_open_min": round(float(m["rmsd_open"].min()), 3),
            "rmsd_open_mean": round(float(m["rmsd_open"].mean()), 3),
            "rmsd_closed_min": round(float(m["rmsd_closed"].min()), 3),
            "rmsd_closed_mean": round(float(m["rmsd_closed"].mean()), 3),
            "occupied_bins": occ,
            "coverage_vs_ns": {str(c): v for c, v in zip(cuts, cov_t)},
            "frac_quad": {k: round(v, 4) for k, v in frac_quad.items()},
            "frac_rmsd": {k: round(v, 4) for k, v in frac_rmsd.items()},
            "n_quad_transitions": n_transitions(quad),
            "n_rmsd_transitions": n_transitions(rmsd_state),
            "cv_pearson": round(corr, 3),
            "mixed_quad_frac": round(mixed, 4),
            "rmsf_mean_A": round(float(rmsf[key].mean()), 3),
            "rmsf_lid_A": round(float(rmsf[key][np.isin(ca_res, list(LID_RES))].mean()), 3),
            "rmsf_nmp_A": round(float(rmsf[key][np.isin(ca_res, list(NMP_RES))].mean()), 3),
            "rmsf_core_A": round(float(rmsf[key][np.isin(ca_res, list(CORE_RES))].mean()), 3),
            "hit_closed_ns": hit_closed_rmsd,
            "hit_open_ns": hit_open_rmsd,
        }
        summary["methods"][key] = rec
        method_rows.append(rec)

    # FES from unbiased histogram and MetaD grid
    def hist_fes(d_nmp, d_lid):
        edges = np.linspace(CV_LO, CV_HI, N_BINS + 1)
        H, xed, yed = np.histogram2d(d_nmp, d_lid, bins=[edges, edges])
        P = H.astype(float)
        P = P / P.sum() if P.sum() else P
        F = np.full_like(P, np.nan)
        mask = P > 0
        F[mask] = -KT_KJ * np.log(P[mask])
        if np.isfinite(F).any():
            F = F - np.nanmin(F)
        return F.T, xed, yed  # transpose so x=NMP, y=LID for pcolormesh

    fes_A, xed, yed = hist_fes(data["A_unbiased"]["d_nmp"], data["A_unbiased"]["d_lid"])
    fes_B_hist, _, _ = hist_fes(data["B_metad"]["d_nmp"], data["B_metad"]["d_lid"])
    fes_path = ROOT / "output" / "fes.npy"
    fes_meta = np.load(fes_path) if fes_path.is_file() else None
    if fes_meta is not None:
        fes_meta = fes_meta - np.nanmin(fes_meta)

    # Approximate open/closed ΔG from unbiased histogram around crystal bins
    def basin_dg(F, xed, yed, d_nmp, d_lid, radius=0.15):
        xcent = 0.5 * (xed[:-1] + xed[1:])
        ycent = 0.5 * (yed[:-1] + yed[1:])
        # F is (n_lid, n_nmp) after transpose
        ix = int(np.argmin(np.abs(xcent - d_nmp)))
        iy = int(np.argmin(np.abs(ycent - d_lid)))
        nx = int(round(radius / BIN))
        slx = slice(max(ix - nx, 0), ix + nx + 1)
        sly = slice(max(iy - nx, 0), iy + nx + 1)
        patch = F[sly, slx]
        if not np.isfinite(patch).any():
            return None
        return float(np.nanmin(patch))

    dg_open = basin_dg(fes_A, xed, yed, crystal_cv["open"]["d_nmp"], crystal_cv["open"]["d_lid"])
    dg_closed = basin_dg(fes_A, xed, yed, crystal_cv["closed"]["d_nmp"], crystal_cv["closed"]["d_lid"])
    summary["unbiased_fes_kJ"] = {
        "open_basin": None if dg_open is None else round(dg_open, 2),
        "closed_basin": None if dg_closed is None else round(dg_closed, 2),
        "deltaG_closed_minus_open": (
            None if (dg_open is None or dg_closed is None) else round(dg_closed - dg_open, 2)
        ),
        "comment": (
            "Histogram FES from unbiased MD occupancy; empty bins are unseen, not infinite. "
            "If closed basin is NaN/absent, 300 ns unbiased MD did not visit closed-like CVs."
        ),
    }
    if fes_meta is not None:
        # OpenMM grid is (n_cv0, n_cv1) = (NMP, LID) on [0.6, 4.5]
        ngrid = fes_meta.shape[0]
        axis = np.linspace(0.6, 4.5, ngrid)
        def meta_at(d_nmp, d_lid):
            ix = int(np.clip(np.argmin(np.abs(axis - d_nmp)), 0, ngrid - 1))
            iy = int(np.clip(np.argmin(np.abs(axis - d_lid)), 0, ngrid - 1))
            # try both index orders
            v1 = fes_meta[ix, iy]
            v2 = fes_meta[iy, ix]
            return float(min(v1, v2))
        summary["metad_fes_kJ"] = {
            "open_crystal": round(meta_at(crystal_cv["open"]["d_nmp"], crystal_cv["open"]["d_lid"]), 2),
            "closed_crystal": round(meta_at(crystal_cv["closed"]["d_nmp"], crystal_cv["closed"]["d_lid"]), 2),
            "grid_min": 0.0,
            "grid_p95": round(float(np.nanpercentile(fes_meta, 95)), 2),
        }
        summary["metad_fes_kJ"]["deltaG_closed_minus_open"] = round(
            summary["metad_fes_kJ"]["closed_crystal"] - summary["metad_fes_kJ"]["open_crystal"], 2
        )

    # TMD first time LID vs NMP cross midpoint
    tmd = data["C_tmd"]
    tmd_lid_close = np.where(tmd["d_lid"] < mid_lid)[0]
    tmd_nmp_close = np.where(tmd["d_nmp"] < mid_nmp)[0]
    summary["tmd_path"] = {
        "first_LID_closed_ns": None if len(tmd_lid_close) == 0 else float(tmd["t"][tmd_lid_close[0]]),
        "first_NMP_closed_ns": None if len(tmd_nmp_close) == 0 else float(tmd["t"][tmd_nmp_close[0]]),
        "final_rmsd_closed_nm": float(tmd["rmsd_closed"][-1]),
        "min_rmsd_closed_nm": float(tmd["rmsd_closed"].min()),
        "initial_rmsd_closed_nm": float(tmd["rmsd_closed"][0]),
    }
    a = summary["tmd_path"]["first_LID_closed_ns"]
    b = summary["tmd_path"]["first_NMP_closed_ns"]
    if a is not None and b is not None:
        if abs(a - b) < 2.0:
            order = "near-concerted"
        elif a < b:
            order = "LID-first"
        else:
            order = "NMP-first"
        summary["tmd_path"]["domain_order"] = order

    # Simulated tempering ladder occupancy (column "State" in temperature.log)
    tlog = ROOT / "output_stemper" / "temperature.log"
    ladder = ROOT / "output_stemper" / "ladder.json"
    if tlog.is_file():
        states = []
        with tlog.open() as handle:
            next(handle)
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        states.append(int(parts[2]))
                    except ValueError:
                        continue
        if states:
            counts = np.bincount(np.asarray(states), minlength=8).astype(float)
            frac = counts / counts.sum()
            temps = json.loads(ladder.read_text())["temps_K"] if ladder.is_file() else list(range(8))
            summary["stemper_ladder"] = {
                "n_samples": int(len(states)),
                "frac_by_state": [round(float(x), 4) for x in frac],
                "temps_K": [round(float(x), 1) for x in temps],
                "frac_Tmin": round(float(frac[0]), 4),
                "frac_Tmax": round(float(frac[-1]), 4),
            }

    # BioEmu generated models before MD relaxation (6 CA frames)
    bioemu_pdbs = sorted((ROOT / "output_bioemu" / "models" / "frames").glob("bioemu_*.pdb"))
    if bioemu_pdbs:
        n_ca = len(idx["all"])
        xyzs = []
        for path in bioemu_pdbs:
            traj = md.load_pdb(str(path))
            cas = [a.index for a in traj.topology.atoms if a.name == "CA"]
            if len(cas) != n_ca:
                print(f"skip {path.name}: {len(cas)} CA vs {n_ca}", flush=True)
                continue
            xyzs.append(traj.atom_slice(cas).xyz[0])
        if xyzs:
            gen = metrics_from_xyz(np.stack(xyzs, axis=0))
            summary["bioemu_generated"] = {
                "n_models": int(len(xyzs)),
                "d_nmp": [round(float(x), 3) for x in gen["d_nmp"]],
                "d_lid": [round(float(x), 3) for x in gen["d_lid"]],
                "rmsd_open_A": [round(float(x) * 10, 2) for x in gen["rmsd_open"]],
                "rmsd_closed_A": [round(float(x) * 10, 2) for x in gen["rmsd_closed"]],
                "min_rmsd_closed_A": round(float(gen["rmsd_closed"].min()) * 10, 2),
                "n_closed_like": int(np.sum(gen["rmsd_closed"] < CLOSED_RMSD_CUT)),
            }

    # Radar scores 0-1
    occs = [summary["methods"][k]["occupied_bins"] for k in methods]
    rmsfs = [summary["methods"][k]["rmsf_mean_A"] for k in methods]
    speeds_v = [summary["methods"][k]["ns_per_day"] or 0 for k in methods]
    closed_frac = [summary["methods"][k]["frac_rmsd"]["closed"] for k in methods]
    open_frac = [summary["methods"][k]["frac_rmsd"]["open"] for k in methods]
    mixed = [summary["methods"][k]["mixed_quad_frac"] for k in methods]
    radar = {}
    for i, key in enumerate(methods):
        rec = summary["methods"][key]
        both_crystals = float(rec["rmsd_open_min"] < 0.45) + float(rec["rmsd_closed_min"] < 0.45)
        radar[key] = {
            "coverage": rec["occupied_bins"] / max(occs),
            "flexibility": rec["rmsf_mean_A"] / max(rmsfs),
            "reach_open": max(0.0, 1.0 - rec["rmsd_open_min"] / crystal_cv["closed"]["rmsd_open"]),
            "reach_closed": max(0.0, 1.0 - rec["rmsd_closed_min"] / crystal_cv["open"]["rmsd_closed"]),
            "efficiency": (rec["ns_per_day"] or 0) / max(speeds_v),
            "path_info": mixed[i] / max(max(mixed), 1e-6),
        }
    summary["radar"] = radar

    knowledge = {
        "A_unbiased": "none (apo MD)",
        "B_metad": "functional CVs (LID/NMP–CORE COM)",
        "C_tmd": "closed crystal 1AKE",
        "D_ensemble": "open + closed crystals (interpolation)",
        "E_stemper": "none (T ladder 300–400 K)",
        "F_gamd": "none (dihedral boost)",
        "G_blind": "generic CVs (Rg + RMSD to start)",
        "H_wrong": "wrong CVs (intra-CORE distances)",
        "I_bioemu": "sequence only, then unbiased MD",
    }
    ranking = []
    for key, rec in summary["methods"].items():
        rec["knowledge"] = knowledge.get(key, "")
        ranking.append({
            "key": key,
            "title": rec["title"],
            "reach_closed_A": round(rec["rmsd_closed_min"] * 10, 2),
            "frac_closed": rec["frac_rmsd"]["closed"],
            "occupied_bins": rec["occupied_bins"],
            "overopen_lid_nm": rec["d_lid_max"],
            "gpu_hours": rec["gpu_hours_eq"],
            "knowledge": rec["knowledge"],
        })
    ranking.sort(key=lambda r: (r["reach_closed_A"], -r["frac_closed"]))
    summary["comparison"] = {
        "axes": [
            "Reach closed: min Cα RMSD to 1AKE and fraction of frames < 3.5 Å",
            "Coverage of functional CV plane (NMP–CORE vs LID–CORE occupied bins)",
            "Over-opening: max LID–CORE vs crystal open 3.08 nm",
            "Knowledge required (sequence / CVs / closed crystal)",
            "Path information: mixed OC/CO quadrants and TMD domain order",
            "Efficiency: ns/day and equivalent GPU hours for ~300 ns",
        ],
        "ranked_by_closed_rmsd": ranking,
    }

    # canvas payload
    rng = np.random.default_rng(0)

    def sample_idx(n, k=SAMPLE_N):
        if n <= k:
            return np.arange(n)
        return np.sort(rng.choice(n, size=k, replace=False))

    canvas = {
        "crystal": crystal_cv,
        "pca_var": [round(float(x) * 100, 1) for x in var_ratio],
        "time_cats": TIME_CATS,
        "coverage_cuts": cuts,
        "methods": {},
        "rmsf_res": ca_res.tolist(),
        "fes_meta_shape": None if fes_meta is None else list(fes_meta.shape),
    }
    for key, spec in methods.items():
        m = data[key]
        n = len(m["t"])
        si = sample_idx(n)
        # time series every 5 ns using actual t for A-C; concatenated for D
        t_use = m["t"] if key not in concat_keys else np.arange(n) * DT_NS
        canvas["methods"][key] = {
            "title": spec["title"],
            "metrics": summary["methods"][key],
            "radar": radar[key],
            "cv_sample": {
                "d_nmp": [round(float(x), 3) for x in m["d_nmp"][si]],
                "d_lid": [round(float(x), 3) for x in m["d_lid"][si]],
            },
            "pca_sample": {
                "pc1": [round(float(x), 3) for x in m["pc1"][si]],
                "pc2": [round(float(x), 3) for x in m["pc2"][si]],
            },
            "rmsd_open_vs_t": downsample_xy(t_use, m["rmsd_open"] * 10, TIME_CATS),  # Angstrom
            "rmsd_closed_vs_t": downsample_xy(t_use, m["rmsd_closed"] * 10, TIME_CATS),
            "d_lid_vs_t": downsample_xy(t_use, m["d_lid"], TIME_CATS),
            "d_nmp_vs_t": downsample_xy(t_use, m["d_nmp"], TIME_CATS),
            "coverage_vs_ns": summary["methods"][key]["coverage_vs_ns"],
            "rmsf_A": [round(float(x), 3) for x in rmsf[key]],
            "quad_frac": summary["methods"][key]["frac_quad"],
            "rmsd_frac": summary["methods"][key]["frac_rmsd"],
        }
    canvas["pca_crystal"] = {
        "open": [round(float(pc_open[0]), 3), round(float(pc_open[1]), 3)],
        "closed": [round(float(pc_closed[0]), 3), round(float(pc_closed[1]), 3)],
    }

    # subsample FES meta as 24x24 for optional later use
    if fes_meta is not None:
        step = max(1, fes_meta.shape[0] // 24)
        sub = fes_meta[::step, ::step]
        canvas["fes_meta_sub"] = np.round(np.clip(sub, 0, 40), 1).tolist()

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "canvas_data.json").write_text(json.dumps(canvas) + "\n")

    with (OUT / "method_metrics.csv").open("w", newline="") as handle:
        fields = [
            "key", "title", "sampled_ns", "n_frames", "ns_per_day", "gpu_hours_eq",
            "occupied_bins", "rmsd_open_min", "rmsd_closed_min", "hit_closed_ns",
            "rmsf_mean_A", "frac_open", "frac_closed", "frac_intermediate",
            "cv_pearson", "n_quad_transitions", "mixed_quad_frac",
        ]
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for key in methods:
            r = summary["methods"][key]
            w.writerow({
                "key": key,
                "title": r["title"],
                "sampled_ns": r["sampled_ns"],
                "n_frames": r["n_frames"],
                "ns_per_day": r["ns_per_day"],
                "gpu_hours_eq": r["gpu_hours_eq"],
                "occupied_bins": r["occupied_bins"],
                "rmsd_open_min": r["rmsd_open_min"],
                "rmsd_closed_min": r["rmsd_closed_min"],
                "hit_closed_ns": r["hit_closed_ns"],
                "rmsf_mean_A": r["rmsf_mean_A"],
                "frac_open": r["frac_rmsd"]["open"],
                "frac_closed": r["frac_rmsd"]["closed"],
                "frac_intermediate": r["frac_rmsd"]["intermediate"],
                "cv_pearson": r["cv_pearson"],
                "n_quad_transitions": r["n_quad_transitions"],
                "mixed_quad_frac": r["mixed_quad_frac"],
            })

    # ---------------- figures ----------------
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    })
    colors = {k: methods[k]["color"] for k in methods}

    # 1. CV map
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 10.0), sharex=True, sharey=True)
    keys = list(methods)
    for ax, key in zip(axes.ravel(), keys):
        m = data[key]
        ax.hexbin(m["d_nmp"], m["d_lid"], gridsize=36, cmap="viridis", mincnt=1, linewidths=0)
        ax.scatter(crystal_cv["open"]["d_nmp"], crystal_cv["open"]["d_lid"], c="white",
                   edgecolors="black", s=28, zorder=3, label="4AKE open")
        ax.scatter(crystal_cv["closed"]["d_nmp"], crystal_cv["closed"]["d_lid"], c="red",
                   edgecolors="black", s=28, zorder=3, label="1AKE closed")
        ax.axvline(mid_nmp, color="0.5", ls="--", lw=0.8)
        ax.axhline(mid_lid, color="0.5", ls="--", lw=0.8)
        ax.set_title(methods[key]["title"], fontsize=9)
        ax.set_xlim(CV_LO, CV_HI)
        ax.set_ylim(CV_LO, CV_HI)
    for ax in axes[-1, :]:
        ax.set_xlabel("NMP–CORE (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("LID–CORE (nm)")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("AdK domain-distance occupancy (100 ps frames)")
    fig.tight_layout()
    fig.savefig(FIG / "cv_hexbin.png")
    plt.close(fig)

    # 2. PCA
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for key in methods:
        m = data[key]
        si = sample_idx(len(m["t"]), 400)
        ax.scatter(m["pc1"][si], m["pc2"][si], s=6, alpha=0.35, c=colors[key], label=methods[key]["title"], linewidths=0)
    ax.scatter(pc_open[0], pc_open[1], c="white", edgecolors="black", s=70, zorder=4, label="4AKE")
    ax.scatter(pc_closed[0], pc_closed[1], c="red", edgecolors="black", s=70, zorder=4, label="1AKE")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title("Joint CA PCA after CORE superposition")
    ax.legend(markerscale=2, frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pca.png")
    plt.close(fig)

    # 3. RMSD vs time
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
    for key in methods:
        m = data[key]
        t_use = m["t"] if key not in concat_keys else np.arange(len(m["t"])) * DT_NS
        st = max(1, len(t_use) // 600)
        axes[0].plot(t_use[::st], m["rmsd_open"][::st] * 10, color=colors[key], lw=0.9, label=methods[key]["title"])
        axes[1].plot(t_use[::st], m["rmsd_closed"][::st] * 10, color=colors[key], lw=0.9, label=methods[key]["title"])
    axes[0].set_ylabel("Cα RMSD to 4AKE (Å)")
    axes[1].set_ylabel("Cα RMSD to 1AKE (Å)")
    axes[1].set_xlabel("Time (ns); D and I are concatenated seeds")
    axes[0].set_title("Global Cα RMSD to experimental open / closed")
    axes[0].legend(frameon=False, ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "rmsd_time.png")
    plt.close(fig)

    # 4. FES
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharex=True, sharey=True)
    im0 = axes[0].pcolormesh(xed, yed, np.ma.masked_invalid(fes_A), cmap="magma_r", shading="auto", vmin=0, vmax=12)
    axes[0].set_title("A unbiased  −kT ln P  (kJ/mol)")
    if fes_meta is not None:
        axis = np.linspace(0.6, 4.5, fes_meta.shape[0])
        # OpenMM stores (cv0=NMP, cv1=LID); pcolormesh wants LID on y
        im1 = axes[1].pcolormesh(axis, axis, fes_meta.T, cmap="magma_r", shading="auto", vmin=0, vmax=40)
        axes[1].set_title("B WT-MetaD  getFreeEnergy (kJ/mol)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="kJ/mol")
    else:
        im1 = None
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="kJ/mol")
    for ax in axes:
        ax.scatter(crystal_cv["open"]["d_nmp"], crystal_cv["open"]["d_lid"], c="white", edgecolors="k", s=35, zorder=3)
        ax.scatter(crystal_cv["closed"]["d_nmp"], crystal_cv["closed"]["d_lid"], c="cyan", edgecolors="k", s=35, zorder=3)
        ax.set_xlabel("NMP–CORE (nm)")
        ax.set_xlim(0.8, 4.4)
        ax.set_ylim(0.8, 4.4)
    axes[0].set_ylabel("LID–CORE (nm)")
    fig.tight_layout()
    fig.savefig(FIG / "fes.png")
    plt.close(fig)

    # 5. RMSF
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = ca_res
    for key in methods:
        ax.plot(x, rmsf[key], color=colors[key], lw=1.1, label=methods[key]["title"])
    ax.axvspan(30, 59, color="0.90", lw=0)
    ax.axvspan(122, 159, color="0.90", lw=0)
    ax.text(44.5, 0.15, "NMP", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(140.5, 0.15, "LID", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Residue")
    ax.set_ylabel("Cα RMSF (Å)")
    ax.set_title("Cα RMSF after CORE superposition to 4AKE")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rmsf.png")
    plt.close(fig)

    # 6. coverage vs time
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key in methods:
        xs = [50, 100, 200, 300]
        ys = [summary["methods"][key]["coverage_vs_ns"][str(c)] for c in xs]
        ax.plot(xs, ys, marker="o", color=colors[key], label=methods[key]["title"])
    ax.set_xlabel("Cumulative sampled time (ns)")
    ax.set_ylabel(f"Occupied {BIN*10:.1f} Å bins in (dNMP, dLID)")
    ax.set_title("Coverage growth at 50 / 100 / 200 / 300 ns")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_time.png")
    plt.close(fig)

    # 7. state bars
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    labels = [methods[k]["title"] for k in methods]
    x = np.arange(len(labels))
    w = 0.22
    for i, s in enumerate(("open", "closed", "intermediate")):
        ys = [summary["methods"][k]["frac_rmsd"][s] * 100 for k in methods]
        ax.bar(x + (i - 1) * w, ys, width=w, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Frame fraction (%)")
    ax.set_title(f"RMSD states  (open < {OPEN_RMSD_CUT*10:.0f} Å to 4AKE, closed < {CLOSED_RMSD_CUT*10:.0f} Å to 1AKE)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rmsd_states.png")
    plt.close(fig)

    print("wrote", OUT)
    print(json.dumps({k: summary["methods"][k] for k in methods}, indent=2)[:4000])


if __name__ == "__main__":
    main()
