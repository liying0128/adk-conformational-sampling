#!/usr/bin/env python3
"""Methods B–D for apo AdK, each 300 ns, run sequentially after baseline.

Method B: well-tempered metadynamics on LID–CORE and NMP–CORE distances
Method C: targeted MD toward closed 1AKE (CA RMSD)
Method D: ensemble MD relaxation from open→closed interpolations
          (drop AF/BioEmu models in inputs/ai_ensemble/*.pdb to replace seeds)

Same GPU lock as baseline, so the two commands can be launched together
without sharing the 4090. This script waits for output/prod.done, then
runs B → C → D with no idle gap.

    python -u run_baseline.py
    python -u run_others.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_baseline as rb  # noqa: E402

CLOSED_PDB_ID = "1AKE"
CLOSED_PDB = rb.INPUTS / "1AKE_protein.pdb"
AI_ENSEMBLE_DIR = rb.INPUTS / "ai_ensemble"

# E. coli AdK domains (PDB numbering)
NMP_RES = range(30, 60)
LID_RES = range(122, 160)
CORE_RES = list(range(1, 30)) + list(range(60, 122)) + list(range(160, 215))

META_BIAS_FACTOR = 12.0
META_HEIGHT_KJ = 1.0
META_FREQ_STEPS = 500          # 2 ps at 4 fs
META_SAVE_STEPS = 25000        # 100 ps
META_SIGMA_NM = 0.05
META_MIN_NM = 0.6
META_MAX_NM = 4.5
META_GRID = 180

TMD_K = 8000.0                 # kJ/mol/nm^2 on CA RMSD
TMD_TARGET_NM = 0.05
TMD_ANNEAL_FRAC = 2.0 / 3.0    # 200 ns pull + 100 ns hold at 300 ns

N_SEEDS = 6


def log(msg: str) -> None:
    print(msg, flush=True)


def residue_ca_atoms(topology, residue_ids):
    wanted = {int(x) for x in residue_ids}
    return [
        atom.index
        for atom in topology.atoms()
        if atom.name == "CA"
        and not rb.is_solvent_residue(atom.residue)
        and int(atom.residue.id) in wanted
    ]


def ca_index_by_resid(topology) -> dict[int, int]:
    out = {}
    for atom in topology.atoms():
        if atom.name != "CA" or rb.is_solvent_residue(atom.residue):
            continue
        out[int(atom.residue.id)] = atom.index
    return out


def com_distance_force(group_a, group_b):
    from openmm import CustomCentroidBondForce

    force = CustomCentroidBondForce(2, "distance(g1,g2)")
    force.addGroup(list(group_a))
    force.addGroup(list(group_b))
    force.addBond([0, 1])
    return force


def download_closed_pdb() -> Path:
    rb.INPUTS.mkdir(parents=True, exist_ok=True)
    if CLOSED_PDB.is_file() and CLOSED_PDB.stat().st_size > 1000:
        return CLOSED_PDB
    url = f"https://files.rcsb.org/download/{CLOSED_PDB_ID}.pdb"
    raw = rb.INPUTS / f"{CLOSED_PDB_ID}.pdb"
    log(f"[prep] downloading {url}")
    urllib.request.urlretrieve(url, raw)
    keep = []
    for line in raw.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == "A":
            keep.append(line)
    CLOSED_PDB.write_text("\n".join(keep) + "\n")
    log(f"[prep] wrote apo chain A to {CLOSED_PDB.name} ({len(keep)} atoms)")
    return CLOSED_PDB


def parse_pdb_ca_nm(path: Path, chain: str = "A") -> dict[int, np.ndarray]:
    coords = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) > 21 and line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        resid = int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) * 0.1
        coords[resid] = xyz
    if coords and min(coords) == 0 and 0 in coords:
        # BioEmu / mdtraj often number residues from 0; OpenMM AdK is 1–214.
        coords = {k + 1: v for k, v in coords.items()}
    return coords


def kabsch(P: np.ndarray, Q: np.ndarray):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def aligned_closed_ca(topology, open_pos_nm: np.ndarray) -> dict[int, np.ndarray]:
    download_closed_pdb()
    closed = parse_pdb_ca_nm(CLOSED_PDB)
    open_map = ca_index_by_resid(topology)
    common = sorted(set(closed) & set(open_map))
    if len(common) < 50:
        raise RuntimeError(f"Too few shared CA atoms for 1AKE vs system: {len(common)}")
    P = np.array([closed[i] for i in common])
    Q = np.array([open_pos_nm[open_map[i]] for i in common])
    R, t = kabsch(P, Q)
    return {resid: R @ xyz + t for resid, xyz in closed.items()}


def positions_nm(simulation) -> np.ndarray:
    from openmm import unit

    return np.array(simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))


def set_positions_nm(simulation, pos_nm: np.ndarray) -> None:
    from openmm import unit

    simulation.context.setPositions(pos_nm * unit.nanometer)


def morph_positions(open_pos: np.ndarray, topology, closed_ca: dict[int, np.ndarray], lam: float) -> np.ndarray:
    pos = open_pos.copy()
    ca_map = ca_index_by_resid(topology)
    for resid, idx in ca_map.items():
        if resid not in closed_ca:
            continue
        pos[idx] = (1.0 - lam) * open_pos[idx] + lam * closed_ca[resid]
    # move other atoms in the residue with their CA
    ca_shift = {}
    for resid, idx in ca_map.items():
        ca_shift[resid] = pos[idx] - open_pos[idx]
    for atom in topology.atoms():
        if rb.is_solvent_residue(atom.residue) or atom.name == "CA":
            continue
        resid = int(atom.residue.id)
        if resid in ca_shift:
            pos[atom.index] = open_pos[atom.index] + ca_shift[resid]
    return pos


def load_open_positions():
    """CPU copy of the equilibrated protein+solvent coordinates."""
    from openmm import Platform
    from openmm.app import Simulation

    pdb, system = rb.load_prepared()
    sim = Simulation(
        pdb.topology,
        system,
        rb.make_integrator(),
        Platform.getPlatformByName("CPU"),
        {"Threads": "2"},
    )
    try:
        copy_equilibrated_state(sim, pdb.topology)
        return pdb.topology, positions_nm(sim)
    finally:
        del sim


def copy_equilibrated_state(dst_sim, topology) -> None:
    pdb, system = rb.load_prepared()
    chk = rb.ROOT / "output" / "eq_npt.chk"
    if not chk.is_file():
        chk = rb.ROOT / "output" / "eq_nvt.chk"
    if not chk.is_file():
        raise FileNotFoundError("Need output/eq_npt.chk (or eq_nvt.chk) from baseline equilibration")
    src = rb.make_simulation(topology, system, force_cpu=False)[0]
    try:
        rb.load_checkpoint(src, chk, topology, system)
        rb.copy_context_state(src, dst_sim)
    finally:
        del src
    dst_sim.currentStep = 0
    log(f"[start] copied equilibrated coordinates from {chk.name}")


def attach_reporters(simulation, total_steps: int, topology, append: bool, label: str) -> None:
    prot_idx = rb.protein_atom_indices(topology)
    prot_top = rb.subset_topology(topology, prot_idx)
    simulation.reporters = []
    rb.add_state_reporters(simulation, rb.PROD_CSV, rb.PROD_LOG, total_steps, append=append)
    simulation.reporters.append(
        rb.LiveProgressReporter(
            label,
            total_steps,
            start_step=simulation.currentStep,
            time_unit="prod",
        )
    )
    simulation.reporters.append(
        rb.SubsetXTCReporter(
            rb.PROD_XTC,
            max(1, rb.ps_to_steps(rb.TRAJ_INTERVAL_PS, rb.DT_FS)),
            prot_idx,
            prot_top,
            append=append,
        )
    )


def production_chunks(simulation, total_steps: int, step_fn, on_chunk=None, fes_fn=None) -> None:
    chunk_steps = max(1, rb.ns_to_steps(rb.CHUNK_NS, rb.DT_FS))
    chk_steps = max(1, rb.ns_to_steps(rb.CHECKPOINT_INTERVAL_NS, rb.DT_FS))
    last_chk = simulation.currentStep
    try:
        while simulation.currentStep < total_steps:
            if on_chunk is not None:
                on_chunk(simulation)
            n = min(chunk_steps, total_steps - simulation.currentStep)
            step_fn(simulation, n)
            if simulation.currentStep - last_chk >= chk_steps or simulation.currentStep >= total_steps:
                rb.save_chk(simulation, rb.PROD_CHK)
                last_chk = simulation.currentStep
                ns_done = simulation.currentStep * rb.DT_FS / 1e6
                ns_total = total_steps * rb.DT_FS / 1e6
                log(f"[chk] {ns_done:.3f} ns / {ns_total:.1f} ns")
                if fes_fn is not None:
                    fes_fn(simulation)
    except KeyboardInterrupt:
        rb.save_chk(simulation, rb.PROD_CHK)
        log(f"\n[chk] interrupted at step {simulation.currentStep}")
        raise
    rb.save_chk(simulation, rb.PROD_CHK)


def wait_for_baseline(production_ns: float) -> None:
    done = rb.ROOT / "output" / "prod.done"
    csv_path = rb.ROOT / "output" / "prod_thermo.csv"
    total_steps = rb.ns_to_steps(production_ns, rb.DT_FS)
    log("[wait] waiting for baseline production to finish (output/prod.done)")
    log("[wait] keep python -u run_baseline.py running in the other terminal")
    t0 = time.time()
    while not done.is_file():
        extra = ""
        if csv_path.is_file():
            try:
                last = csv_path.read_text().strip().splitlines()[-1]
                extra = f" | baseline last log: {last[:120]}"
            except Exception:
                extra = ""
        elapsed = rb.format_duration(time.time() - t0)
        log(f"[wait] still running  elapsed {elapsed}  target {production_ns:.0f} ns ({total_steps} steps){extra}")
        time.sleep(30)
    log(f"[wait] baseline done, reading {done}")


def make_method_simulation(topology, system, force_cpu: bool, total_steps: int):
    simulation, platform, properties, available, n_cpu, n_gpu, details = rb.make_simulation(
        topology, system, force_cpu=force_cpu
    )
    rb.print_banner(
        platform,
        properties,
        available,
        n_cpu,
        n_gpu,
        total_steps,
        total_steps * rb.DT_FS / 1e6,
        details,
    )
    return simulation


def run_metad(args, total_steps: int) -> int:
    log("=" * 72)
    log("Method B: well-tempered metadynamics  CVs = d(NMP-CORE), d(LID-CORE)")
    log("=" * 72)
    rb.bind_output_dir(rb.ROOT / "output_metad")
    pdb, system = rb.load_prepared()
    nmp = residue_ca_atoms(pdb.topology, NMP_RES)
    lid = residue_ca_atoms(pdb.topology, LID_RES)
    core = residue_ca_atoms(pdb.topology, CORE_RES)
    log(f"[metad] CA atoms  NMP={len(nmp)} LID={len(lid)} CORE={len(core)}")

    from openmm.app.metadynamics import BiasVariable, Metadynamics
    from openmm import unit

    cv_nmp = BiasVariable(com_distance_force(nmp, core), META_MIN_NM, META_MAX_NM, META_SIGMA_NM, False, META_GRID)
    cv_lid = BiasVariable(com_distance_force(lid, core), META_MIN_NM, META_MAX_NM, META_SIGMA_NM, False, META_GRID)
    hills = rb.OUTPUT / "hills"
    hills.mkdir(parents=True, exist_ok=True)
    meta = Metadynamics(
        system,
        [cv_nmp, cv_lid],
        rb.TEMPERATURE_K * unit.kelvin,
        META_BIAS_FACTOR,
        META_HEIGHT_KJ * unit.kilojoule_per_mole,
        META_FREQ_STEPS,
        saveFrequency=META_SAVE_STEPS,
        biasDir=str(hills),
    )
    simulation = make_method_simulation(pdb.topology, system, args.cpu, total_steps)

    have_prod = rb.PROD_CHK.is_file() and not args.fresh
    if have_prod:
        rb.load_checkpoint(simulation, rb.PROD_CHK, pdb.topology, system)
        log(f"[metad] resume step={simulation.currentStep}")
    else:
        copy_equilibrated_state(simulation, pdb.topology)
        rb.set_barostat_frequency(simulation, 25)
        rb.save_chk(simulation, rb.PROD_CHK)

    if simulation.currentStep >= total_steps:
        log("[metad] already finished")
        return 0

    append = rb.PROD_XTC.is_file() and simulation.currentStep > 0
    attach_reporters(simulation, total_steps, pdb.topology, append, "metad")
    cv_log = rb.OUTPUT / "cv.csv"
    if not append:
        cv_log.write_text("step,ns,d_nmp_nm,d_lid_nm\n")

    def write_fes(_sim):
        fes = meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
        np.save(rb.OUTPUT / "fes.npy", fes)
        vals = meta.getCollectiveVariables(simulation)
        ns = simulation.currentStep * rb.DT_FS / 1e6
        with cv_log.open("a") as handle:
            handle.write(f"{simulation.currentStep},{ns:.4f},{vals[0]:.4f},{vals[1]:.4f}\n")

    log(f"[metad] production {total_steps * rb.DT_FS / 1e6:.1f} ns")
    production_chunks(simulation, total_steps, meta.step, fes_fn=write_fes)
    write_fes(simulation)
    log("[metad] finished")
    return 0


def run_tmd(args, total_steps: int) -> int:
    log("=" * 72)
    log("Method C: targeted MD toward closed 1AKE (CA RMSD)")
    log("=" * 72)
    rb.bind_output_dir(rb.ROOT / "output_tmd")
    topology, open_pos = load_open_positions()
    closed_ca = aligned_closed_ca(topology, open_pos)
    pdb, system = rb.load_prepared()
    from openmm import CustomCVForce, RMSDForce

    ca_map = ca_index_by_resid(pdb.topology)
    common = sorted(set(ca_map) & set(closed_ca))
    particles = [ca_map[i] for i in common]
    # RMSDForce needs one reference position per System particle, not just the CA subset.
    n_atoms = pdb.topology.getNumAtoms()
    if len(open_pos) != n_atoms:
        raise RuntimeError(f"open positions {len(open_pos)} != system atoms {n_atoms}")
    ref_arr = np.array(open_pos, dtype=float)
    for resid in common:
        ref_arr[ca_map[resid]] = closed_ca[resid]
    log(f"[tmd] RMSD atoms: {len(particles)} CA  (system atoms {n_atoms})")
    rmsd_force = RMSDForce(ref_arr, particles)
    tmd = CustomCVForce("0.5*k*(rmsd-target)^2")
    tmd.addCollectiveVariable("rmsd", rmsd_force)
    tmd.addGlobalParameter("k", TMD_K)
    tmd.addGlobalParameter("target", 0.5)
    system.addForce(tmd)

    simulation = make_method_simulation(pdb.topology, system, args.cpu, total_steps)
    schedule = rb.OUTPUT / "tmd_schedule.json"
    anneal_steps = int(total_steps * TMD_ANNEAL_FRAC)

    have_prod = rb.PROD_CHK.is_file() and not args.fresh
    if have_prod:
        rb.load_checkpoint(simulation, rb.PROD_CHK, pdb.topology, system)
        info = json.loads(schedule.read_text()) if schedule.is_file() else {"rmsd0": 0.8}
        log(f"[tmd] resume step={simulation.currentStep}")
    else:
        copy_equilibrated_state(simulation, pdb.topology)
        rb.set_barostat_frequency(simulation, 25)
        rmsd0 = float(tmd.getCollectiveVariableValues(simulation.context)[0])
        info = {"rmsd0": rmsd0, "target_final": TMD_TARGET_NM, "anneal_steps": anneal_steps}
        schedule.write_text(json.dumps(info, indent=2) + "\n")
        log(f"[tmd] initial CA RMSD to closed = {rmsd0:.3f} nm")
        rb.save_chk(simulation, rb.PROD_CHK)

    if simulation.currentStep >= total_steps:
        log("[tmd] already finished")
        return 0

    rmsd0 = float(info["rmsd0"])

    def set_target(sim):
        step = sim.currentStep
        if anneal_steps <= 0:
            frac = 1.0
        else:
            frac = min(1.0, step / anneal_steps)
        target = rmsd0 * (1.0 - frac) + TMD_TARGET_NM * frac
        sim.context.setParameter("k", TMD_K)
        sim.context.setParameter("target", target)

    append = rb.PROD_XTC.is_file() and simulation.currentStep > 0
    attach_reporters(simulation, total_steps, pdb.topology, append, "tmd")
    log(f"[tmd] production {total_steps * rb.DT_FS / 1e6:.1f} ns")
    production_chunks(simulation, total_steps, lambda sim, n: sim.step(n), on_chunk=set_target)
    log("[tmd] finished")
    return 0


def run_ensemble(args, total_steps: int, out_root: Path | None = None, pdb_dir: Path | None = None) -> int:
    log("=" * 72)
    log("Method D: ensemble MD relaxation, total 300 ns split across seeds")
    log("=" * 72)
    topology, open_pos = load_open_positions()
    closed_ca = aligned_closed_ca(topology, open_pos)

    out_root = Path(out_root) if out_root is not None else (rb.ROOT / "output_ai")
    pdb_dir = Path(pdb_dir) if pdb_dir is not None else AI_ENSEMBLE_DIR
    user_pdbs = sorted(pdb_dir.glob("*.pdb")) if pdb_dir.is_dir() else []
    if user_pdbs:
        log(f"[ensemble] using {len(user_pdbs)} models from {pdb_dir}")
        seeds = user_pdbs[:N_SEEDS]
        lambdas = [None] * len(seeds)
    else:
        lambdas = np.linspace(0.15, 1.0, N_SEEDS).tolist()
        seeds = [None] * N_SEEDS
        log(f"[ensemble] no inputs/ai_ensemble PDBs; using 1AKE interpolations λ={lambdas}")

    n_seed = len(seeds)
    steps_each = total_steps // n_seed
    leftover = total_steps - steps_each * n_seed
    meta = {
        "n_seeds": n_seed,
        "steps_each": steps_each,
        "lambdas": lambdas,
        "user_pdbs": [str(p.name) for p in user_pdbs[:n_seed]],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ensemble.json").write_text(json.dumps(meta, indent=2) + "\n")

    for i, (lam, user_pdb) in enumerate(zip(lambdas, seeds)):
        seed_steps = steps_each + (leftover if i == n_seed - 1 else 0)
        run_dir = out_root / f"seed_{i:02d}"
        rb.bind_output_dir(run_dir)
        pdb, system = rb.load_prepared()
        simulation = make_method_simulation(pdb.topology, system, args.cpu, seed_steps)
        have_prod = rb.PROD_CHK.is_file() and not args.fresh
        if have_prod:
            rb.load_checkpoint(simulation, rb.PROD_CHK, pdb.topology, system)
            pos0 = positions_nm(simulation)
            if simulation.currentStep <= 0 or not np.isfinite(pos0).all():
                log(f"[ensemble {i}] discarding failed checkpoint step={simulation.currentStep}")
                have_prod = False
            else:
                log(f"[ensemble {i}] resume step={simulation.currentStep}/{seed_steps}")
        if not have_prod:
            copy_equilibrated_state(simulation, pdb.topology)
            pos = positions_nm(simulation)
            if user_pdb is not None:
                model_ca = parse_pdb_ca_nm(user_pdb)
                open_map = ca_index_by_resid(pdb.topology)
                common = sorted(set(model_ca) & set(open_map))
                P = np.array([model_ca[j] for j in common])
                Q = np.array([open_pos[open_map[j]] for j in common])
                R, t = kabsch(P, Q)
                aligned = {resid: R @ xyz + t for resid, xyz in model_ca.items()}
                best_pos = None
                last_lam = None
                for lam_try in (0.4, 0.7, 1.0):
                    pos = morph_positions(open_pos, pdb.topology, aligned, lam_try)
                    set_positions_nm(simulation, pos)
                    rb.set_barostat_frequency(simulation, 0)
                    simulation.minimizeEnergy(maxIterations=400 if args.test else 2500)
                    pos = positions_nm(simulation)
                    if np.isfinite(pos).all():
                        best_pos = pos
                        last_lam = lam_try
                    else:
                        log(f"[ensemble {i}] NaN after morph λ={lam_try:.2f}")
                        break
                if best_pos is None:
                    raise RuntimeError(f"NaN coordinates after seeding {user_pdb.name}")
                set_positions_nm(simulation, best_pos)
                log(
                    f"[ensemble {i}] seeded from {user_pdb.name}  "
                    f"matched_CA={len(common)}  λ={last_lam:.2f}"
                )
            else:
                pos = morph_positions(open_pos, pdb.topology, closed_ca, float(lam))
                log(f"[ensemble {i}] interpolation λ={lam:.2f}")
                set_positions_nm(simulation, pos)
                rb.set_barostat_frequency(simulation, 0)
                simulation.minimizeEnergy(maxIterations=400 if args.test else 2000)
            simulation.context.setVelocitiesToTemperature(rb.TEMPERATURE_K)
            rb.set_barostat_frequency(simulation, 25)
            simulation.currentStep = 0
            rb.save_chk(simulation, rb.PROD_CHK)

        if simulation.currentStep >= seed_steps:
            log(f"[ensemble {i}] already finished")
            continue
        append = rb.PROD_XTC.is_file() and simulation.currentStep > 0
        attach_reporters(simulation, seed_steps, pdb.topology, append, f"ens{i}")
        production_chunks(simulation, seed_steps, lambda sim, n: sim.step(n))
        log(f"[ensemble {i}] finished {seed_steps * rb.DT_FS / 1e6:.1f} ns")
        del simulation
    log("[ensemble] all seeds finished (300 ns total)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AdK methods B–D, 300 ns each")
    parser.add_argument("--test", action="store_true", help="Short pipeline check")
    parser.add_argument("--fresh", action="store_true", help="Ignore method checkpoints")
    parser.add_argument("--cpu", action="store_true", help="Force CPU platform")
    parser.add_argument("--nowait", action="store_true", help="Do not wait for baseline prod.done")
    parser.add_argument(
        "--only",
        default="metad,tmd,ensemble",
        help="Comma list: metad,tmd,ensemble",
    )
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.chdir(ROOT)
    methods = [m.strip() for m in args.only.split(",") if m.strip()]
    production_ns = rb.PRODUCTION_NS
    total_steps = rb.ns_to_steps(production_ns, rb.DT_FS)
    if args.test:
        total_steps = 200
        production_ns = total_steps * rb.DT_FS / 1e6
        args.nowait = True
        log("[test] 200 steps per method")

    log("AdK methods B–D")
    log(f"each method {production_ns:.1f} ns  ({total_steps} steps)  order={methods}")
    download_closed_pdb()

    if not args.nowait:
        wait_for_baseline(rb.PRODUCTION_NS)

    if args.fresh:
        for folder in ("output_metad", "output_tmd", "output_ai"):
            path = ROOT / folder
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix in {".chk", ".xtc", ".csv", ".log", ".npy", ".done"}:
                        child.unlink()

    runners = {
        "metad": run_metad,
        "tmd": run_tmd,
        "ensemble": run_ensemble,
    }
    with rb.GpuLock(rb.ROOT / "output" / "gpu.lock"):
        for name in methods:
            if name not in runners:
                log(f"[skip] unknown method {name}")
                continue
            try:
                rc = runners[name](args, total_steps)
            except KeyboardInterrupt:
                return 130
            except Exception:
                traceback.print_exc()
                return 1
            if rc not in (0, None):
                return rc
    log("All requested methods finished.")
    done = rb.ROOT / "output" / "others.done"
    done.write_text(json.dumps({"finished": True, "time": time.strftime("%F %T")}, indent=2) + "\n")
    log(f"wrote {done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
