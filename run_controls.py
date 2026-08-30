#!/usr/bin/env python3
"""Control experiments after A–D: sampling with less (or wrong) prior knowledge.

Each method is 300 ns production, sequential, same GPU lock as A–D.

  E  simulated tempering   no CV, T = 300–400 K (Tmax)
  F  dihedral GaMD         no CV, no target structure
  G  blind WT-MetaD        only the starting structure (Rg + RMSD-to-start)
  H  wrong-CV WT-MetaD     two intra-CORE distances that do not describe opening

Does not use 1AKE except that the shared solvated box was built from 4AKE.

    python -u run_controls.py --test          # short GPU smoke test
    python -u run_controls.py --nowait        # full 300 ns each, after A–D
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_baseline as rb  # noqa: E402
import run_others as ro  # noqa: E402

TMIN_K = 300.0
TMAX_K = 400.0
N_TEMPS = 8
ST_ITER_PS = 2.0
GAMD_STATS_NS = 2.0
GAMD_SIGMA0_KT = 6.0
META_BIAS_FACTOR = 12.0
META_HEIGHT_KJ = 1.0
META_FREQ_STEPS = 500
META_SAVE_STEPS = 25000
META_GRID = 160

# Intra-CORE pairs: both sides are CORE, not LID/NMP hinges.
WRONG_A = range(8, 19)
WRONG_B = range(90, 101)
WRONG_C = range(170, 186)
WRONG_D = range(200, 215)

CONTROLS_DONE = ROOT / "output" / "controls.done"


def log(msg: str) -> None:
    print(msg, flush=True)


def all_ca(topology):
    return [
        atom.index
        for atom in topology.atoms()
        if atom.name == "CA" and not rb.is_solvent_residue(atom.residue)
    ]


def clone_system():
    pdb, system = rb.load_prepared()
    return pdb, system


def method_dir(args, name: str) -> Path:
    if args.test:
        return ROOT / "output_test_controls" / name
    return ROOT / f"output_{name}"
    done = ROOT / "output" / "others.done"
    log("[wait] waiting for methods B–D (output/others.done)")
    t0 = time.time()
    while not done.is_file():
        log(f"[wait] others still running  elapsed {rb.format_duration(time.time() - t0)}")
        time.sleep(30)
    log(f"[wait] others done, reading {done}")


def start_from_eq(simulation, topology) -> None:
    ro.copy_equilibrated_state(simulation, topology)
    rb.set_barostat_frequency(simulation, 25)


def extract_torsion_force(system):
    from openmm import PeriodicTorsionForce, XmlSerializer

    idx = None
    force = None
    for i, f in enumerate(system.getForces()):
        if isinstance(f, PeriodicTorsionForce):
            idx = i
            force = f
            break
    if force is None or idx is None:
        raise RuntimeError("System has no PeriodicTorsionForce (needed for GaMD dihedral boost)")
    clone = XmlSerializer.deserialize(XmlSerializer.serialize(force))
    system.removeForce(idx)
    return clone


def add_dihedral_gamd(system, E_kj: float, k_per_kj: float):
    from openmm import CustomCVForce

    torsion = extract_torsion_force(system)
    boost = CustomCVForce("dih + 0.5*k*delta*delta*step(delta); delta=E-dih")
    boost.addCollectiveVariable("dih", torsion)
    boost.addGlobalParameter("E", E_kj)
    boost.addGlobalParameter("k", k_per_kj)
    system.addForce(boost)
    return boost


def run_stemper(args, total_steps: int) -> int:
    from openmm import unit
    from openmm.app import ExpandedEnsembleSampler

    log("=" * 72)
    log(f"Method E: simulated tempering  T={TMIN_K:.0f}–{TMAX_K:.0f} K  (no CV, no 1AKE)")
    log("=" * 72)
    run_dir = method_dir(args, "stemper")
    rb.bind_output_dir(run_dir)
    pdb, system = clone_system()
    simulation = ro.make_method_simulation(pdb.topology, system, args.cpu, total_steps)

    sampler_chk = run_dir / "sampler.pkl"
    have_prod = rb.PROD_CHK.is_file() and not args.fresh
    if have_prod:
        rb.load_checkpoint(simulation, rb.PROD_CHK, pdb.topology, system)
        log(f"[stemper] resume OpenMM step={simulation.currentStep}")
    else:
        start_from_eq(simulation, pdb.topology)
        rb.save_chk(simulation, rb.PROD_CHK)

    if simulation.currentStep >= total_steps:
        log("[stemper] already finished")
        return 0

    append = rb.PROD_XTC.is_file() and simulation.currentStep > 0
    ro.attach_reporters(simulation, total_steps, pdb.topology, append, "stemper")

    iter_steps = max(1, rb.ps_to_steps(ST_ITER_PS, rb.DT_FS))
    report_steps = max(iter_steps, rb.ps_to_steps(rb.TRAJ_INTERVAL_PS, rb.DT_FS))
    if args.test:
        iter_steps = 10
        report_steps = 20
    temps = np.geomspace(TMIN_K, TMAX_K, N_TEMPS)
    states = [{"temperature": float(t) * unit.kelvin} for t in temps]
    log(f"[stemper] temperatures K = {[round(float(t), 1) for t in temps]}")
    log(f"[stemper] exchange every {iter_steps} steps, log every {report_steps} steps")
    resume = have_prod and sampler_chk.is_file() and not args.fresh
    sampler = ExpandedEnsembleSampler(
        states,
        simulation,
        iter_steps,
        reportInterval=report_steps,
        logFile=str(run_dir / "temperature.log"),
        checkpointFile=str(sampler_chk),
        resume=resume,
    )
    (run_dir / "ladder.json").write_text(
        json.dumps({"Tmin": TMIN_K, "Tmax": TMAX_K, "temps_K": [float(t) for t in temps]}, indent=2)
        + "\n"
    )
    log(f"[stemper] production {total_steps * rb.DT_FS / 1e6:.1f} ns")
    ro.production_chunks(simulation, total_steps, lambda sim, n: sim.step(n))
    log("[stemper] finished")
    del sampler
    return 0


def collect_dihedral_stats(simulation, n_steps: int, sample_steps: int) -> dict:
    from openmm import PeriodicTorsionForce, unit

    group = 7
    torsion = None
    for force in simulation.system.getForces():
        if isinstance(force, PeriodicTorsionForce):
            torsion = force
            break
    if torsion is None:
        raise RuntimeError("no PeriodicTorsionForce for GaMD statistics")
    torsion.setForceGroup(group)
    simulation.context.reinitialize(preserveState=True)
    vals = []
    done = 0
    while done < n_steps:
        n = min(sample_steps, n_steps - done)
        simulation.step(n)
        done += n
        e = simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
        vals.append(e.value_in_unit(unit.kilojoule_per_mole))
    arr = np.asarray(vals, dtype=float)
    kT = 0.008314462618 * rb.TEMPERATURE_K
    sigma0 = GAMD_SIGMA0_KT * kT
    vmin = float(arr.min())
    vmax = float(arr.max())
    vavg = float(arr.mean())
    sigma = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    energy_thr = vmax
    denom = energy_thr - vmin
    if denom <= 1e-6 or sigma <= 1e-8:
        k0 = 0.0
    else:
        k0 = min(1.0, (sigma0 / sigma) * (energy_thr - vavg) / denom)
    k = k0 / denom if denom > 1e-6 else 0.0
    params = {
        "V_min": vmin,
        "V_max": vmax,
        "V_avg": vavg,
        "sigma": sigma,
        "E": energy_thr,
        "k0": k0,
        "k": k,
        "n_samples": int(len(arr)),
        "stats_ns": n_steps * rb.DT_FS / 1e6,
    }
    log(
        f"[gamd] dihedral stats  Vmin={vmin:.1f} Vmax={vmax:.1f} "
        f"Vavg={vavg:.1f} σ={sigma:.1f} kJ/mol  k0={k0:.3f} k={k:.6f}"
    )
    return params


def run_gamd(args, total_steps: int) -> int:
    log("=" * 72)
    log("Method F: dihedral GaMD  (no CV, no target structure)")
    log("=" * 72)
    run_dir = method_dir(args, "gamd")
    rb.bind_output_dir(run_dir)
    param_path = run_dir / "gamd_params.json"

    have_prod = rb.PROD_CHK.is_file() and param_path.is_file() and not args.fresh
    if have_prod:
        params = json.loads(param_path.read_text())
        pdb, system = clone_system()
        add_dihedral_gamd(system, params["E"], params["k"])
        simulation = ro.make_method_simulation(pdb.topology, system, args.cpu, total_steps)
        rb.load_checkpoint(simulation, rb.PROD_CHK, pdb.topology, system)
        log(f"[gamd] resume step={simulation.currentStep}")
    else:
        pdb, system = clone_system()
        stat_sim = ro.make_method_simulation(pdb.topology, system, args.cpu, total_steps)
        start_from_eq(stat_sim, pdb.topology)
        stats_steps = rb.ns_to_steps(GAMD_STATS_NS, rb.DT_FS)
        sample = max(1, rb.ps_to_steps(1.0, rb.DT_FS))
        if args.test:
            stats_steps = 40
            sample = 5
        log(f"[gamd] collecting dihedral statistics for {stats_steps * rb.DT_FS / 1e6:.3f} ns")
        params = collect_dihedral_stats(stat_sim, stats_steps, sample)
        positions = stat_sim.context.getState(getPositions=True).getPositions()
        velocities = stat_sim.context.getState(getVelocities=True).getVelocities()
        box = stat_sim.context.getState().getPeriodicBoxVectors()
        del stat_sim
        run_dir.mkdir(parents=True, exist_ok=True)
        param_path.write_text(json.dumps(params, indent=2) + "\n")

        pdb, system = clone_system()
        add_dihedral_gamd(system, params["E"], params["k"])
        simulation = ro.make_method_simulation(pdb.topology, system, args.cpu, total_steps)
        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)
        simulation.context.setPeriodicBoxVectors(*box)
        rb.set_barostat_frequency(simulation, 25)
        simulation.currentStep = 0
        rb.save_chk(simulation, rb.PROD_CHK)

    if simulation.currentStep >= total_steps:
        log("[gamd] already finished")
        return 0
    append = rb.PROD_XTC.is_file() and simulation.currentStep > 0
    ro.attach_reporters(simulation, total_steps, pdb.topology, append, "gamd")
    log(f"[gamd] production {total_steps * rb.DT_FS / 1e6:.1f} ns")
    ro.production_chunks(simulation, total_steps, lambda sim, n: sim.step(n))
    log("[gamd] finished")
    return 0


def run_cv_metad(args, total_steps: int, label: str, run_dir: Path, make_cvs) -> int:
    from openmm.app.metadynamics import Metadynamics
    from openmm import unit

    log("=" * 72)
    log(label)
    log("=" * 72)
    rb.bind_output_dir(run_dir)
    pdb, system = clone_system()
    cvs = make_cvs(pdb, system)
    hills = run_dir / "hills"
    hills.mkdir(parents=True, exist_ok=True)
    meta = Metadynamics(
        system,
        cvs,
        rb.TEMPERATURE_K * unit.kelvin,
        META_BIAS_FACTOR,
        META_HEIGHT_KJ * unit.kilojoule_per_mole,
        META_FREQ_STEPS if not args.test else 20,
        saveFrequency=META_SAVE_STEPS if not args.test else 40,
        biasDir=str(hills),
    )
    simulation = ro.make_method_simulation(pdb.topology, system, args.cpu, total_steps)
    have_prod = rb.PROD_CHK.is_file() and not args.fresh
    if have_prod:
        rb.load_checkpoint(simulation, rb.PROD_CHK, pdb.topology, system)
        log(f"[{run_dir.name}] resume step={simulation.currentStep}")
    else:
        start_from_eq(simulation, pdb.topology)
        rb.save_chk(simulation, rb.PROD_CHK)

    if simulation.currentStep >= total_steps:
        log(f"[{run_dir.name}] already finished")
        return 0

    append = rb.PROD_XTC.is_file() and simulation.currentStep > 0
    ro.attach_reporters(simulation, total_steps, pdb.topology, append, run_dir.name)
    cv_log = run_dir / "cv.csv"
    if not append:
        cv_log.write_text("step,ns," + ",".join(f"cv{i}" for i in range(len(cvs))) + "\n")

    def write_fes(_sim):
        fes = meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
        np.save(run_dir / "fes.npy", fes)
        vals = meta.getCollectiveVariables(simulation)
        ns = simulation.currentStep * rb.DT_FS / 1e6
        with cv_log.open("a") as handle:
            handle.write(
                f"{simulation.currentStep},{ns:.4f}," + ",".join(f"{v:.4f}" for v in vals) + "\n"
            )

    log(f"[{run_dir.name}] production {total_steps * rb.DT_FS / 1e6:.1f} ns")
    ro.production_chunks(simulation, total_steps, meta.step, fes_fn=write_fes)
    write_fes(simulation)
    log(f"[{run_dir.name}] finished")
    return 0


def make_blind_cvs(pdb, system):
    from openmm import RGForce, RMSDForce
    from openmm.app.metadynamics import BiasVariable

    ca = all_ca(pdb.topology)
    rg = RGForce(ca)
    _, eq_pos = ro.load_open_positions()
    if len(eq_pos) != system.getNumParticles():
        raise RuntimeError(f"eq positions {len(eq_pos)} != system atoms {system.getNumParticles()}")
    rmsd = RMSDForce(eq_pos, ca)
    log(f"[blind] CVs: Rg({len(ca)} CA) and RMSD to equilibrated starting CA")
    return [
        BiasVariable(rg, 1.20, 2.80, 0.03, False, META_GRID),
        BiasVariable(rmsd, 0.00, 1.50, 0.04, False, META_GRID),
    ]


def make_wrong_cvs(pdb, _system):
    from openmm.app.metadynamics import BiasVariable

    a = ro.residue_ca_atoms(pdb.topology, WRONG_A)
    b = ro.residue_ca_atoms(pdb.topology, WRONG_B)
    c = ro.residue_ca_atoms(pdb.topology, WRONG_C)
    d = ro.residue_ca_atoms(pdb.topology, WRONG_D)
    log(f"[wrong] CORE–CORE CVs  nCA={len(a)}/{len(b)} and {len(c)}/{len(d)}  (not LID/NMP)")
    d1 = ro.com_distance_force(a, b)
    d2 = ro.com_distance_force(c, d)
    return [
        BiasVariable(d1, 0.40, 2.40, 0.04, False, META_GRID),
        BiasVariable(d2, 0.40, 2.40, 0.04, False, META_GRID),
    ]


def run_blind(args, total_steps: int) -> int:
    return run_cv_metad(
        args,
        total_steps,
        "Method G: blind WT-MetaD  CVs = Rg(CA), RMSD to starting CA  (no 1AKE, no domain map)",
        method_dir(args, "blind"),
        make_blind_cvs,
    )


def run_wrong(args, total_steps: int) -> int:
    return run_cv_metad(
        args,
        total_steps,
        "Method H: wrong-CV WT-MetaD  CVs = two intra-CORE distances  (known-bad CVs)",
        method_dir(args, "wrong"),
        make_wrong_cvs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AdK knowledge-level control MD, 300 ns each")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nowait", action="store_true")
    parser.add_argument(
        "--only",
        default="stemper,gamd,blind,wrong",
        help="Comma list: stemper,gamd,blind,wrong",
    )
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.chdir(ROOT)
    rb.configure_thread_env(rb.cpu_count())
    methods = [m.strip() for m in args.only.split(",") if m.strip()]
    production_ns = rb.PRODUCTION_NS
    total_steps = rb.ns_to_steps(production_ns, rb.DT_FS)
    if args.test:
        total_steps = 200
        production_ns = total_steps * rb.DT_FS / 1e6
        args.nowait = True
        log("[test] 200 production steps per method")

    log("AdK knowledge-level controls")
    log(f"each method {production_ns:.1f} ns  ({total_steps} steps)  order={methods}")
    log(f"CPU threads={rb.cpu_count()}  GPU lock={ROOT / 'output' / 'gpu.lock'}")

    if not args.nowait:
        wait_for_others()

    if args.fresh:
        for folder in ("output_stemper", "output_gamd", "output_blind", "output_wrong", "output_test_controls"):
            path = ROOT / folder
            if not path.is_dir():
                continue
            for child in path.rglob("*"):
                if child.is_file() and child.suffix in {".chk", ".xtc", ".csv", ".log", ".npy", ".pkl", ".done"}:
                    child.unlink()

    runners = {
        "stemper": run_stemper,
        "gamd": run_gamd,
        "blind": run_blind,
        "wrong": run_wrong,
    }
    with rb.GpuLock(ROOT / "output" / "gpu.lock"):
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
    log("All requested control methods finished.")
    if not args.test:
        CONTROLS_DONE.write_text(
            json.dumps({"finished": True, "time": time.strftime("%F %T"), "methods": methods}, indent=2)
            + "\n"
        )
        log(f"wrote {CONTROLS_DONE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
