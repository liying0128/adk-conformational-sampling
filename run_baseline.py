#!/usr/bin/env python3
"""Apo E. coli AdK (4AKE) unbiased MD baseline.

All paths are relative to this script directory. Copy the whole
`baseline_unbiased_md` folder to the GPU server and run:

    conda install -c conda-forge openmm pdbfixer
    python -u run_baseline.py

Uses the GPU (CUDA if present, otherwise OpenCL) for nonbonded work and
all CPU threads for PME. Force CPU with --cpu.

    nohup python -u run_baseline.py > output/console.log 2>&1 &
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Simulation settings (edit here if needed)
# ---------------------------------------------------------------------------
PRODUCTION_NS = 300.0          # 300 ns production, not counting min/eq
DT_FS = 4.0                    # 4 fs with hydrogen-mass repartitioning
TEMPERATURE_K = 300.0
PRESSURE_BAR = 1.0
PADDING_NM = 1.2
IONIC_STRENGTH_M = 0.15
HYDROGEN_MASS_AMU = 1.5
NONBONDED_CUTOFF_NM = 1.0
FRICTION_PER_PS = 1.0
NVT_NS = 0.2
NPT_NS = 1.0
TRAJ_INTERVAL_PS = 100.0       # protein-only XTC
LOG_INTERVAL_PS = 1000.0
STDOUT_INTERVAL_PS = 10000.0
CHECKPOINT_INTERVAL_NS = 10.0
CHUNK_NS = 10.0
PROGRESS_INTERVAL_S = 5.0      # wall-clock refresh for live progress
FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3p.xml")
WATER_RESIDUES = {"HOH", "WAT", "TIP3", "SOL"}
ION_RESIDUES = {"NA", "CL", "Na+", "Cl-", "NA+", "CL-", "SOD", "CLA", "K", "MG", "Na", "Cl"}

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
PROTEIN_PDB = INPUTS / "4AKE_protein.pdb"

OUTPUT = ROOT / "output"
SOLVATED_PDB = OUTPUT / "solvated.pdb"
PROTEIN_ONLY_PDB = OUTPUT / "protein.pdb"
SYSTEM_XML = OUTPUT / "system.xml"
RUN_INFO = OUTPUT / "run_info.json"
MIN_CHK = OUTPUT / "minimized.chk"
NVT_CHK = OUTPUT / "eq_nvt.chk"
NPT_CHK = OUTPUT / "eq_npt.chk"
PROD_CHK = OUTPUT / "prod.chk"
PROD_XTC = OUTPUT / "prod_protein.xtc"
PROD_LOG = OUTPUT / "prod.log"
PROD_CSV = OUTPUT / "prod_thermo.csv"
PROD_DONE = OUTPUT / "prod.done"
GPU_LOCK = OUTPUT / "gpu.lock"


def bind_output_dir(run_dir: Path) -> None:
    """Prepared system always lives in output/; run artifacts can go to output_test/."""
    global OUTPUT, SOLVATED_PDB, PROTEIN_ONLY_PDB, SYSTEM_XML, RUN_INFO
    global MIN_CHK, NVT_CHK, NPT_CHK, PROD_CHK, PROD_XTC, PROD_LOG, PROD_CSV, PROD_DONE, GPU_LOCK
    OUTPUT = ROOT / "output"
    SOLVATED_PDB = OUTPUT / "solvated.pdb"
    PROTEIN_ONLY_PDB = OUTPUT / "protein.pdb"
    SYSTEM_XML = OUTPUT / "system.xml"
    RUN_INFO = OUTPUT / "run_info.json"
    MIN_CHK = run_dir / "minimized.chk"
    NVT_CHK = run_dir / "eq_nvt.chk"
    NPT_CHK = run_dir / "eq_npt.chk"
    PROD_CHK = run_dir / "prod.chk"
    PROD_XTC = run_dir / "prod_protein.xtc"
    PROD_LOG = run_dir / "prod.log"
    PROD_CSV = run_dir / "prod_thermo.csv"
    PROD_DONE = run_dir / "prod.done"
    GPU_LOCK = ROOT / "output" / "gpu.lock"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)


def configure_thread_env(n_cpu: int) -> None:
    n = str(max(1, n_cpu))
    for key in (
        "OPENMM_CPU_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[key] = n


def cpu_count() -> int:
    return os.cpu_count() or 1


class GpuLock:
    """Exclusive file lock so baseline and the other methods do not share one GPU."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fp = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a+")
        print(f"[gpu] waiting for lock {self.path} ...", flush=True)
        fcntl.flock(self._fp, fcntl.LOCK_EX)
        self._fp.seek(0)
        self._fp.truncate()
        self._fp.write(f"pid={os.getpid()} time={time.strftime('%F %T')}\n")
        self._fp.flush()
        print(f"[gpu] acquired lock (pid={os.getpid()})", flush=True)
        return self

    def __exit__(self, *exc):
        if self._fp is not None:
            try:
                fcntl.flock(self._fp, fcntl.LOCK_UN)
            finally:
                self._fp.close()
                self._fp = None
            print("[gpu] released lock", flush=True)
        return False


def write_prod_done(path: Path, total_steps: int, production_ns: float) -> None:
    info = {
        "finished": True,
        "production_ns": production_ns,
        "total_steps": total_steps,
        "time": time.strftime("%F %T"),
        "pid": os.getpid(),
    }
    path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"[prod] wrote {path}", flush=True)


def nvidia_gpu_count() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))


def fs_to_ps(dt_fs: float) -> float:
    return dt_fs * 0.001


def ns_to_steps(ns: float, dt_fs: float) -> int:
    return int(round((ns * 1_000_000.0) / dt_fs))


def us_to_steps(us: float, dt_fs: float) -> int:
    return ns_to_steps(us * 1000.0, dt_fs)


def ps_to_steps(ps: float, dt_fs: float) -> int:
    return int(round((ps * 1000.0) / dt_fs))


def is_solvent_residue(residue) -> bool:
    name = residue.name.strip()
    return name in WATER_RESIDUES or name in ION_RESIDUES


def protein_atom_indices(topology):
    return [atom.index for atom in topology.atoms() if not is_solvent_residue(atom.residue)]


def subset_topology(topology, atom_indices):
    from openmm import app

    index_set = set(atom_indices)
    new_top = app.Topology()
    box = topology.getPeriodicBoxVectors()
    if box is not None:
        new_top.setPeriodicBoxVectors(box)
    old_to_new = {}
    for chain in topology.chains():
        new_chain = None
        for residue in chain.residues():
            keep_atoms = [atom for atom in residue.atoms() if atom.index in index_set]
            if not keep_atoms:
                continue
            if new_chain is None:
                new_chain = new_top.addChain(chain.id)
            new_res = new_top.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            for atom in keep_atoms:
                old_to_new[atom.index] = new_top.addAtom(atom.name, atom.element, new_res, atom.id)
    for bond in topology.bonds():
        a, b = bond
        if a.index in old_to_new and b.index in old_to_new:
            new_top.addBond(old_to_new[a.index], old_to_new[b.index])
    return new_top


class SubsetXTCReporter:
    def __init__(self, filename, report_interval, atom_indices, topology_subset, append=False):
        self._file_name = str(filename)
        self._report_interval = int(report_interval)
        self._atom_indices = list(atom_indices)
        self._topology = topology_subset
        self._append = append
        self._xtc = None
        if not append:
            Path(self._file_name).parent.mkdir(parents=True, exist_ok=True)
            Path(self._file_name).write_bytes(b"")

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return {"steps": steps, "periodic": True, "include": ["positions"]}

    def report(self, simulation, state):
        from openmm.app import XTCFile

        if self._xtc is None:
            self._xtc = XTCFile(
                self._file_name,
                self._topology,
                simulation.integrator.getStepSize(),
                simulation.currentStep,
                self._report_interval,
                self._append,
            )
        positions = state.getPositions(asNumpy=True)
        self._xtc.writeModel(
            positions[self._atom_indices],
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )


def format_duration(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def stdout_is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def print_progress_line(text: str, *, overwrite: bool) -> None:
    if overwrite:
        width = max(88, len(text) + 2)
        sys.stdout.write("\r" + text.ljust(width))
        sys.stdout.flush()
    else:
        print(text, flush=True)


class LiveProgressReporter:
    """Print simulation progress every few wall-clock seconds."""

    def __init__(
        self,
        label: str,
        total_steps: int,
        start_step: int = 0,
        interval_s: float = PROGRESS_INTERVAL_S,
        time_unit: str = "ns",
    ):
        self.label = label
        self.total_steps = max(1, int(total_steps))
        self.start_step = int(start_step)
        self.interval_s = max(0.5, float(interval_s))
        self.time_unit = time_unit
        self.t0 = time.time()
        self.last_print = 0.0
        self._overwrite = stdout_is_tty()
        self._print_state(start_step, None, force=True)

    def describeNextReport(self, simulation):
        remaining = self.total_steps - simulation.currentStep
        if remaining <= 0:
            steps = 1
        else:
            elapsed = time.time() - self.t0
            done = max(0, simulation.currentStep - self.start_step)
            if elapsed > 1.0 and done > 0:
                steps = int(max(1, round((done / elapsed) * self.interval_s)))
            else:
                steps = 50
            steps = max(1, min(steps, remaining))
        return {"steps": steps, "periodic": False, "include": ["energy"]}

    def report(self, simulation, state):
        energy = None
        try:
            from openmm import unit

            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        except Exception:
            energy = None
        finished = simulation.currentStep >= self.total_steps
        self._print_state(simulation.currentStep, energy, force=finished)
        if finished and self._overwrite:
            print(flush=True)

    def _print_state(self, current_step: int, energy, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_print) < self.interval_s and current_step < self.total_steps:
            return
        self.last_print = now
        elapsed = now - self.t0
        current_step = min(max(current_step, self.start_step), self.total_steps)
        session_steps = max(0, current_step - self.start_step)
        speed = (session_steps * DT_FS / 1e6 / elapsed) * 86400 if elapsed > 0 and session_steps > 0 else 0.0
        remaining_steps = max(0, self.total_steps - current_step)
        eta = (remaining_steps / session_steps) * elapsed if session_steps > 0 else None
        if self.time_unit in ("prod", "us", "abs"):
            frac = current_step / max(1, self.total_steps)
            sim_ns = current_step * DT_FS / 1e6
            total_ns = self.total_steps * DT_FS / 1e6
            done_s = f"{sim_ns:.3f}/{total_ns:.1f} ns"
        else:
            span = max(1, self.total_steps - self.start_step)
            frac = session_steps / span
            sim_ns = session_steps * DT_FS / 1e6
            total_ns = span * DT_FS / 1e6
            done_s = f"{sim_ns:.4f}/{total_ns:.4f} ns"
        bar_n = 20
        filled = int(round(bar_n * frac))
        bar = "#" * filled + "-" * (bar_n - filled)
        energy_s = f"  E={energy:.0f} kJ/mol" if energy is not None else ""
        line = (
            f"[{self.label}] {bar} {100.0 * frac:6.2f}%  {done_s}  "
            f"{speed:7.1f} ns/day{energy_s}  "
            f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta) if eta is not None else '--'}"
        )
        print_progress_line(line, overwrite=self._overwrite)


def write_pdb(path: Path, topology, positions) -> None:
    from openmm.app import PDBFile

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)


def prepare_system():
    import numpy as np
    from openmm import MonteCarloBarostat, XmlSerializer, unit
    from openmm import app
    from pdbfixer import PDBFixer

    if not PROTEIN_PDB.is_file():
        raise FileNotFoundError(f"Missing protein PDB: {PROTEIN_PDB}")

    print(f"[prepare] loading {PROTEIN_PDB.relative_to(ROOT)}")
    fixer = PDBFixer(filename=str(PROTEIN_PDB))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    modeller = app.Modeller(fixer.topology, fixer.positions)
    forcefield = app.ForceField(*FORCEFIELD_FILES)
    print("[prepare] solvating (dodecahedron, 1.2 nm padding, 150 mM NaCl)")
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=PADDING_NM * unit.nanometer,
        boxShape="dodecahedron",
        ionicStrength=IONIC_STRENGTH_M * unit.molar,
        neutralize=True,
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=HYDROGEN_MASS_AMU * unit.amu,
        ewaldErrorTolerance=1.0e-5,
    )
    system.addForce(
        MonteCarloBarostat(
            PRESSURE_BAR * unit.bar,
            TEMPERATURE_K * unit.kelvin,
            25,
        )
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_pdb(SOLVATED_PDB, modeller.topology, modeller.positions)
    prot_idx = protein_atom_indices(modeller.topology)
    prot_top = subset_topology(modeller.topology, prot_idx)
    pos_nm = np.array(modeller.positions.value_in_unit(unit.nanometer))
    prot_pos = unit.Quantity(pos_nm[np.array(prot_idx, dtype=int)], unit.nanometer)
    write_pdb(PROTEIN_ONLY_PDB, prot_top, prot_pos)
    SYSTEM_XML.write_text(XmlSerializer.serialize(system))

    n_atoms = modeller.topology.getNumAtoms()
    n_protein = len(prot_idx)
    n_water = sum(1 for res in modeller.topology.residues() if res.name in WATER_RESIDUES)
    info = {
        "protein_pdb": str(PROTEIN_PDB.relative_to(ROOT)),
        "n_atoms": n_atoms,
        "n_protein_atoms": n_protein,
        "n_water": n_water,
        "forcefield": list(FORCEFIELD_FILES),
        "dt_fs": DT_FS,
        "production_ns": PRODUCTION_NS,
        "temperature_K": TEMPERATURE_K,
        "pressure_bar": PRESSURE_BAR,
        "padding_nm": PADDING_NM,
        "ionic_strength_M": IONIC_STRENGTH_M,
        "hydrogen_mass_amu": HYDROGEN_MASS_AMU,
    }
    RUN_INFO.write_text(json.dumps(info, indent=2) + "\n")
    print(f"[prepare] atoms={n_atoms} protein_atoms={n_protein} waters={n_water}")
    print(f"[prepare] wrote {SOLVATED_PDB.relative_to(ROOT)} and {SYSTEM_XML.relative_to(ROOT)}")
    return info


def load_prepared():
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    if not SOLVATED_PDB.is_file() or not SYSTEM_XML.is_file():
        prepare_system()
    pdb = PDBFile(str(SOLVATED_PDB))
    system = XmlSerializer.deserialize(SYSTEM_XML.read_text())
    return pdb, system


def get_barostat(system):
    from openmm import MonteCarloBarostat

    for force in system.getForces():
        if isinstance(force, MonteCarloBarostat):
            return force
    return None


def make_integrator():
    from openmm import LangevinMiddleIntegrator, unit

    return LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        fs_to_ps(DT_FS) * unit.picoseconds,
    )


def context_properties(simulation) -> dict:
    plat = simulation.context.getPlatform()
    info = {"platform": plat.getName()}
    try:
        names = plat.getPropertyNames()
    except Exception:
        names = ()
    for name in names:
        try:
            info[name] = plat.getPropertyValue(simulation.context, name)
        except Exception:
            continue
    return info


def copy_context_state(src, dst) -> None:
    kwargs = {
        "getPositions": True,
        "getVelocities": True,
        "getParameters": True,
        "enforcePeriodicBox": False,
    }
    try:
        state = src.context.getState(getIntegratorParameters=True, **kwargs)
    except TypeError:
        state = src.context.getState(**kwargs)
    dst.context.setState(state)
    dst.currentStep = src.currentStep


def load_checkpoint(simulation, path: Path, topology, system) -> None:
    """Load a checkpoint, copying state through another platform if needed."""
    from openmm import Platform
    from openmm.app import Simulation

    path = Path(path)
    plat_name = simulation.context.getPlatform().getName()
    try:
        simulation.loadCheckpoint(str(path))
        print(f"[resume] loaded {path.name} on {plat_name}, step={simulation.currentStep}", flush=True)
        return
    except Exception as exc:
        print(
            f"[resume] {path.name} cannot load on {plat_name} ({exc}); trying other platforms",
            flush=True,
        )

    n_cpu = str(max(1, cpu_count()))
    available = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    tries = []
    if "CUDA" in available:
        tries.append(("CUDA", {"Precision": "mixed", "DeviceIndex": "0"}))
    if "OpenCL" in available:
        tries.append(("OpenCL", {"Precision": "mixed", "DeviceIndex": "0", "UseCpuPme": "true"}))
    if "CPU" in available:
        tries.append(("CPU", {"Threads": n_cpu}))

    last_error = None
    for name, props in tries:
        if name == plat_name:
            continue
        bridge = None
        try:
            bridge = Simulation(
                topology,
                system,
                make_integrator(),
                Platform.getPlatformByName(name),
                props,
            )
            bridge.loadCheckpoint(str(path))
            copy_context_state(bridge, simulation)
            print(
                f"[resume] copied {path.name} via {name} onto {plat_name}, step={simulation.currentStep}",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
            print(f"[resume] {name} bridge failed: {exc}", flush=True)
        finally:
            del bridge
    raise RuntimeError(f"Could not load checkpoint {path}: {last_error}") from last_error


def make_simulation(topology, system, *, force_cpu: bool = False):
    from openmm import Platform
    from openmm.app import Simulation

    n_cpu = cpu_count()
    configure_thread_env(n_cpu)
    n_gpu = nvidia_gpu_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        n_visible = len([x for x in visible.split(",") if x.strip() != ""])
        if n_visible:
            n_gpu = n_visible

    available = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    gpu_index = "0"
    candidates = []
    if not force_cpu:
        if "CUDA" in available:
            cuda_base = {"Precision": "mixed", "DeviceIndex": gpu_index}
            candidates.append(("CUDA", {**cuda_base, "UseCpuPme": "true"}))
            candidates.append(("CUDA", {**cuda_base, "UseCpuPme": "false"}))
            if n_gpu > 1:
                candidates.insert(
                    0,
                    (
                        "CUDA",
                        {
                            **cuda_base,
                            "DeviceIndex": ",".join(str(i) for i in range(n_gpu)),
                            "UseCpuPme": "true",
                        },
                    ),
                )
        if "HIP" in available:
            candidates.append((
                "HIP",
                {
                    "Precision": "mixed",
                    "DeviceIndex": ",".join(str(i) for i in range(max(1, n_gpu))),
                },
            ))
        if "OpenCL" in available:
            ocl_base = {"Precision": "mixed", "DeviceIndex": gpu_index}
            candidates.append(("OpenCL", {**ocl_base, "UseCpuPme": "true"}))
            candidates.append(("OpenCL", {**ocl_base, "UseCpuPme": "false"}))
    if "CPU" in available:
        candidates.append(("CPU", {"Threads": str(n_cpu)}))
    if "Reference" in available:
        candidates.append(("Reference", {}))

    last_error = None
    for name, properties in candidates:
        try:
            platform = Platform.getPlatformByName(name)
            simulation = Simulation(topology, system, make_integrator(), platform, properties)
            details = context_properties(simulation)
            print(f"[platform] using {name} {properties}", flush=True)
            extra = []
            if details.get("DeviceName"):
                extra.append(f"device={details['DeviceName']}")
            if details.get("UseCpuPme"):
                extra.append(f"UseCpuPme={details['UseCpuPme']}")
            extra.append(f"CPU threads={n_cpu}")
            print("[platform] " + "  ".join(extra), flush=True)
            return simulation, platform, properties, available, n_cpu, n_gpu, details
        except Exception as exc:
            last_error = exc
            print(f"[platform] {name} {properties} failed: {exc}", flush=True)
    raise RuntimeError(f"Could not create OpenMM context: {last_error}") from last_error


def set_barostat_frequency(simulation, frequency: int) -> None:
    barostat = get_barostat(simulation.system)
    if barostat is None:
        return
    if barostat.getFrequency() == frequency:
        return
    barostat.setFrequency(frequency)
    simulation.context.reinitialize(preserveState=True)


def save_chk(simulation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    simulation.saveCheckpoint(str(path))


def add_state_reporters(simulation, csv_path: Path, log_path: Path, total_steps: int, append: bool) -> None:
    from openmm.app import StateDataReporter

    log_steps = max(1, ps_to_steps(LOG_INTERVAL_PS, DT_FS))
    stdout_steps = max(1, ps_to_steps(STDOUT_INTERVAL_PS, DT_FS))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    simulation.reporters.append(
        StateDataReporter(
            str(csv_path),
            log_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            remainingTime=True,
            progress=True,
            totalSteps=total_steps,
            append=append,
        )
    )
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            stdout_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            remainingTime=True,
            progress=True,
            totalSteps=total_steps,
        )
    )
    simulation.reporters.append(
        StateDataReporter(
            str(log_path),
            stdout_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            elapsedTime=True,
            append=append,
        )
    )


def run_minimization(simulation, max_iters: int, chk_path: Path) -> None:
    from openmm import unit

    print(f"[min] energy minimization (maxIterations={max_iters})", flush=True)
    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    print(f"[min] starting E={e0:.1f} kJ/mol", flush=True)
    t0 = time.time()
    chunk = 50 if max_iters > 50 else max(1, max_iters)
    done = 0
    last_e = e0
    overwrite = stdout_is_tty()
    while done < max_iters:
        n = min(chunk, max_iters - done)
        simulation.minimizeEnergy(maxIterations=n)
        done += n
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        elapsed = time.time() - t0
        dE = energy - last_e
        line = (
            f"[min] iter {done}/{max_iters}  E={energy:.1f} kJ/mol  "
            f"dE={dE:+.1f}  elapsed {format_duration(elapsed)}"
        )
        print_progress_line(line, overwrite=overwrite)
        last_e = energy
        if abs(dE) < 1.0 and done >= chunk:
            if overwrite:
                print(flush=True)
            print("[min] energy change < 1 kJ/mol, treating as converged", flush=True)
            break
    if overwrite:
        print(flush=True)
    save_chk(simulation, chk_path)
    elapsed = time.time() - t0
    print(
        f"[min] done in {elapsed/60:.1f} min, E={last_e:.1f} kJ/mol, wrote {chk_path.name}",
        flush=True,
    )


def run_stage(simulation, n_steps: int, chk_path: Path, label: str) -> None:
    if n_steps <= 0:
        return
    start = simulation.currentStep
    target = start + n_steps
    print(
        f"[{label}] {n_steps} steps ({n_steps * DT_FS / 1e6:.4f} ns), live progress every {PROGRESS_INTERVAL_S:.0f}s",
        flush=True,
    )
    saved = list(simulation.reporters)
    simulation.reporters = [
        LiveProgressReporter(label, target, start_step=start, time_unit="ns")
    ]
    t0 = time.time()
    try:
        simulation.step(n_steps)
    finally:
        simulation.reporters = saved
        if stdout_is_tty():
            print(flush=True)
    save_chk(simulation, chk_path)
    elapsed = time.time() - t0
    sim_ns = n_steps * DT_FS / 1e6
    speed = (sim_ns / elapsed) * 86400 if elapsed > 0 else 0.0
    print(
        f"[{label}] done in {elapsed/60:.1f} min, ~{speed:.0f} ns/day, wrote {chk_path.name}",
        flush=True,
    )


def production_loop(simulation, total_steps: int) -> None:
    chunk_steps = max(1, ns_to_steps(CHUNK_NS, DT_FS))
    chk_steps = max(1, ns_to_steps(CHECKPOINT_INTERVAL_NS, DT_FS))
    last_chk = simulation.currentStep
    try:
        while simulation.currentStep < total_steps:
            remaining = total_steps - simulation.currentStep
            n = min(chunk_steps, remaining)
            simulation.step(n)
            if simulation.currentStep - last_chk >= chk_steps or simulation.currentStep >= total_steps:
                save_chk(simulation, PROD_CHK)
                last_chk = simulation.currentStep
                ns_done = simulation.currentStep * DT_FS / 1e6
                ns_total = total_steps * DT_FS / 1e6
                print(f"[prod] checkpoint at {ns_done:.3f} ns / {ns_total:.1f} ns", flush=True)
    except KeyboardInterrupt:
        save_chk(simulation, PROD_CHK)
        print(f"\n[prod] interrupted at step {simulation.currentStep}, checkpoint saved")
        raise
    save_chk(simulation, PROD_CHK)


def print_banner(platform, properties, available, n_cpu, n_gpu, total_steps: int, production_ns: float, details=None) -> None:
    print("=" * 72)
    print("AdK apo unbiased baseline MD")
    print(f"script dir : {ROOT}")
    print(f"platform   : {platform.getName()}  properties={properties}")
    if details:
        device = details.get("DeviceName") or details.get("platform")
        pme = details.get("UseCpuPme", "n/a")
        prec = details.get("Precision", "n/a")
        print(f"device     : {device}  precision={prec}  CPU-PME={pme}")
    print(f"available  : {available}")
    print(f"CPU threads: {n_cpu}")
    print(f"GPU count  : {n_gpu}")
    print(f"dt         : {DT_FS} fs")
    print(f"production : {production_ns} ns  ({total_steps} steps)")
    print("=" * 72, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unbiased MD baseline for apo AdK")
    parser.add_argument("--prepare-only", action="store_true", help="Build solvated system and exit")
    parser.add_argument("--test", action="store_true", help="Short run to verify the pipeline")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--cpu", action="store_true", help="Force CPU platform (skip GPU)")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    bind_output_dir(ROOT / ("output_test" if args.test else "output"))
    os.chdir(ROOT)

    if args.fresh:
        for path in (MIN_CHK, NVT_CHK, NPT_CHK, PROD_CHK, PROD_XTC, PROD_CSV, PROD_LOG, PROD_DONE):
            if path.exists():
                path.unlink()

    if args.prepare_only:
        prepare_system()
        return 0

    lock = GpuLock(ROOT / "output" / "gpu.lock")
    with lock:
        return _run_production(args)


def _run_production(args) -> int:

    production_ns = PRODUCTION_NS
    nvt_ns = NVT_NS
    npt_ns = NPT_NS
    total_steps = ns_to_steps(production_ns, DT_FS)
    if args.test:
        nvt_ns = DT_FS * 200 / 1e6
        npt_ns = DT_FS * 200 / 1e6
        total_steps = 200
        production_ns = total_steps * DT_FS / 1e6
        print("[test] short pipeline: min + 200 NVT + 200 NPT + 200 prod steps")

    pdb, system = load_prepared()
    simulation, platform, properties, available, n_cpu, n_gpu, details = make_simulation(
        pdb.topology, system, force_cpu=args.cpu
    )
    print_banner(platform, properties, available, n_cpu, n_gpu, total_steps, production_ns, details)
    prot_idx = protein_atom_indices(pdb.topology)
    prot_top = subset_topology(pdb.topology, prot_idx)

    have_prod = PROD_CHK.is_file() and not args.fresh
    have_npt = NPT_CHK.is_file() and not args.fresh
    have_nvt = NVT_CHK.is_file() and not args.fresh
    have_min = MIN_CHK.is_file() and not args.fresh

    if have_prod:
        print(f"[resume] loading {PROD_CHK.relative_to(ROOT)}")
        load_checkpoint(simulation, PROD_CHK, pdb.topology, system)
        print(f"[resume] current step={simulation.currentStep}")
    else:
        if have_npt:
            print(f"[resume] loading {NPT_CHK.name}")
            load_checkpoint(simulation, NPT_CHK, pdb.topology, system)
        elif have_nvt:
            print(f"[resume] loading {NVT_CHK.name}")
            load_checkpoint(simulation, NVT_CHK, pdb.topology, system)
            set_barostat_frequency(simulation, 25)
            run_stage(simulation, ns_to_steps(npt_ns, DT_FS), NPT_CHK, "npt")
        elif have_min:
            print(f"[resume] loading {MIN_CHK.name}")
            load_checkpoint(simulation, MIN_CHK, pdb.topology, system)
            set_barostat_frequency(simulation, 0)
            run_stage(simulation, ns_to_steps(nvt_ns, DT_FS), NVT_CHK, "nvt")
            set_barostat_frequency(simulation, 25)
            run_stage(simulation, ns_to_steps(npt_ns, DT_FS), NPT_CHK, "npt")
        else:
            simulation.context.setPositions(pdb.positions)
            set_barostat_frequency(simulation, 0)
            min_iters = 200 if args.test else 5000
            run_minimization(simulation, min_iters, MIN_CHK)
            simulation.context.setVelocitiesToTemperature(TEMPERATURE_K)
            run_stage(simulation, ns_to_steps(nvt_ns, DT_FS), NVT_CHK, "nvt")
            set_barostat_frequency(simulation, 25)
            run_stage(simulation, ns_to_steps(npt_ns, DT_FS), NPT_CHK, "npt")
        simulation.currentStep = 0
        save_chk(simulation, PROD_CHK)

    if simulation.currentStep >= total_steps:
        print("[prod] already finished")
        write_prod_done(PROD_DONE, total_steps, production_ns)
        return 0

    set_barostat_frequency(simulation, 25)
    append_traj = PROD_XTC.is_file() and simulation.currentStep > 0
    simulation.reporters = []
    add_state_reporters(simulation, PROD_CSV, PROD_LOG, total_steps, append=append_traj)
    simulation.reporters.append(
        LiveProgressReporter(
            "prod",
            total_steps,
            start_step=simulation.currentStep,
            interval_s=1.0 if args.test else PROGRESS_INTERVAL_S,
            time_unit="prod",
        )
    )
    simulation.reporters.append(
        SubsetXTCReporter(
            PROD_XTC,
            max(1, ps_to_steps(TRAJ_INTERVAL_PS, DT_FS)),
            prot_idx,
            prot_top,
            append=append_traj,
        )
    )

    print(
        f"[prod] starting at step {simulation.currentStep} / {total_steps} "
        f"({simulation.currentStep * DT_FS / 1e6:.3f} / {production_ns:.1f} ns)",
        flush=True,
    )
    t0 = time.time()
    try:
        production_loop(simulation, total_steps)
    except KeyboardInterrupt:
        return 130
    except Exception:
        save_chk(simulation, PROD_CHK)
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    print(f"[prod] finished {production_ns} ns in {elapsed/3600:.2f} h")
    write_prod_done(PROD_DONE, total_steps, production_ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
