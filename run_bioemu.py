#!/usr/bin/env python3
"""Method D (true AI): BioEmu ensemble from sequence, then 300 ns MD relaxation.

Waits for output/controls.done so it does not steal the 4090 from E–H.
Uses the separate bioemu venv for generation and the OpenMM env for MD.

    python -u run_bioemu.py --nowait
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_baseline as rb  # noqa: E402
import run_others as ro  # noqa: E402

def _bioemu_python() -> Path:
    env = os.environ.get("BIOEMU_PYTHON")
    if env:
        return Path(env)
    for cand in (
        Path.home() / "miniconda3/envs/bioemu/bin/python",
        Path.home() / "anaconda3/envs/bioemu/bin/python",
        Path.home() / "mambaforge/envs/bioemu/bin/python",
    ):
        if cand.is_file():
            return cand
    found = shutil.which("python")
    return Path(found or sys.executable)


BIOEMU_PY = _bioemu_python()
FASTA = ROOT / "inputs" / "4AKE.fasta"
MODELS = ROOT / "output_bioemu" / "models"
FRAMES = MODELS / "frames"
OUT_MD = ROOT / "output_bioemu"
DONE = ROOT / "output" / "bioemu.done"
N_SAMPLES = 16  # extra so filtering can still leave N_SEEDS
N_SEEDS = 6


def log(msg: str) -> None:
    print(msg, flush=True)


def wait_flag(path: Path, name: str) -> None:
    log(f"[wait] waiting for {name} ({path})")
    t0 = time.time()
    while not path.is_file():
        log(f"[wait] {name} still running  elapsed {rb.format_duration(time.time() - t0)}")
        time.sleep(30)
    log(f"[wait] {name} done")


def existing_frames() -> list[Path]:
    if not FRAMES.is_dir():
        return []
    return sorted(p for p in FRAMES.glob("bioemu_*.pdb") if p.is_file() and p.stat().st_size > 200)


def extract_frames(n: int) -> list[Path]:
    """Split BioEmu topology.pdb + samples.xtc into chain-A PDB frames."""
    have = existing_frames()
    if len(have) >= n:
        return have[:n]
    xtc = MODELS / "samples.xtc"
    top = MODELS / "topology.pdb"
    if not xtc.is_file() or not top.is_file():
        return have
    FRAMES.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import mdtraj as md\n"
        "xtc, top, outdir, n = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])\n"
        "traj = md.load(xtc, top=top)\n"
        "outdir.mkdir(parents=True, exist_ok=True)\n"
        "n = min(n, traj.n_frames)\n"
        "for i in range(n):\n"
        "    pdb = outdir / f'bioemu_{i:02d}.pdb'\n"
        "    traj[i].save_pdb(str(pdb))\n"
        "    lines = []\n"
        "    for line in pdb.read_text().splitlines():\n"
        "        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 26:\n"
        "            line = line[:21] + 'A' + line[22:]\n"
        "            resid = int(line[22:26])\n"
        "            line = line[:22] + f'{resid + 1:4d}' + line[26:]\n"
        "        lines.append(line)\n"
        "    pdb.write_text('\\n'.join(lines) + '\\n')\n"
        "print(n)\n"
    )
    log(f"[bioemu] extracting {n} PDB frames from {xtc.name}")
    subprocess.check_call([str(BIOEMU_PY), "-c", code, str(xtc), str(top), str(FRAMES), str(n)])
    return existing_frames()[:n]


def generate_bioemu(n: int) -> None:
    if not BIOEMU_PY.is_file():
        raise FileNotFoundError(f"BioEmu python not found: {BIOEMU_PY}")
    if not FASTA.is_file():
        raise FileNotFoundError(f"missing {FASTA}")
    MODELS.mkdir(parents=True, exist_ok=True)
    log(f"[bioemu] generating {n} AdK samples -> {MODELS}")
    cmd = [
        str(BIOEMU_PY),
        "-m",
        "bioemu.sample",
        "--sequence",
        str(FASTA),
        "--num_samples",
        str(n),
        "--output_dir",
        str(MODELS),
    ]
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # ColabFold/AlphaFold embeddings go through JAX. This machine's JAX CUDA
    # build cannot initialize cuDNN (dnn_support != nullptr); CPU is enough
    # for 214-residue embeddings. BioEmu sampling itself uses PyTorch CUDA.
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLATFORM_NAME"] = "cpu"
    log("[bioemu] " + " ".join(cmd))
    log(f"[bioemu] HF_ENDPOINT={env.get('HF_ENDPOINT')} JAX_PLATFORMS={env.get('JAX_PLATFORMS')}")
    subprocess.check_call(cmd, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--nowait", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.chdir(ROOT)
    rb.configure_thread_env(rb.cpu_count())

    if not args.nowait and not args.test:
        wait_flag(ROOT / "output" / "controls.done", "controls E-H")

    n_gen = 2 if args.test else N_SAMPLES
    need = 1 if args.test else N_SEEDS
    if args.test:
        args.nowait = True

    def ensure_models() -> list[Path]:
        if args.skip_generate:
            return extract_frames(need)
        if args.fresh and MODELS.is_dir():
            shutil.rmtree(MODELS)
        have = extract_frames(need)
        if len(have) < need:
            generate_bioemu(n_gen)
            have = extract_frames(need)
        else:
            log(f"[bioemu] reusing {len(have)} existing models")
        if len(have) < need:
            raise RuntimeError(f"BioEmu produced {len(have)} PDBs, need {need}")
        return have

    if args.test:
        ensure_models()
        log("[test] generation check only, skip 300 ns MD")
        return 0

    class EnsembleArgs:
        test = False
        fresh = args.fresh
        cpu = args.cpu

    total_steps = rb.ns_to_steps(rb.PRODUCTION_NS, rb.DT_FS)
    with rb.GpuLock(ROOT / "output" / "gpu.lock"):
        pdbs = ensure_models()
        log(f"[bioemu-md] 300 ns relaxation from {len(pdbs)} models in {FRAMES}")
        ro.run_ensemble(EnsembleArgs(), total_steps, out_root=OUT_MD, pdb_dir=FRAMES)
    DONE.write_text(json.dumps({"finished": True, "time": time.strftime("%F %T")}, indent=2) + "\n")
    log(f"wrote {DONE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
