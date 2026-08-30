# Apo AdK conformational sampling benchmark

OpenMM scripts, analysis code, and numerical results for:

> Zhao, Q.; Cui, T.; Li, Y. *Comparative Conformational Sampling of Apo Adenylate Kinase: From Classical Enhanced Molecular Dynamics to Generative AI Ensembles.*

Qingling Zhao and Ting Cui contributed equally. Correspondence: [liying0128@buu.edu.cn](mailto:liying0128@buu.edu.cn).

This repository holds **code, inputs, and analysis tables/figures**. Production trajectories (protein-only XTC, ~40 MB each) are not stored here; they are available from the corresponding author on reasonable request.

## System

- Protein: apo *E. coli* adenylate kinase, PDB [4AKE](https://www.rcsb.org/structure/4AKE) chain A (214 residues), no ligand
- Closed reference: ligand-stripped [1AKE](https://www.rcsb.org/structure/1AKE) chain A
- Force field: Amber ff14SB + TIP3P (`amber14-all.xml`, `amber14/tip3p.xml`)
- Production: **300 ns per method**, 4 fs timestep (HMR), 300 K, 1 bar, OpenMM

## Methods (each 300 ns)

| ID | Script | Protocol |
|----|--------|----------|
| A | `run_baseline.py` | Unbiased MD |
| B | `run_others.py` | Well-tempered metadynamics on LID–CORE and NMP–CORE Cα COM distances |
| C | `run_others.py` | Targeted MD toward 1AKE (Cα RMSD restraint) |
| D | `run_others.py` | Six open→closed interpolations × 50 ns unbiased MD |
| E | `run_controls.py` | Simulated tempering, 300–400 K, 8 rungs |
| F | `run_controls.py` | Dihedral GaMD |
| G | `run_controls.py` | Blind WT-MetaD (Rg + RMSD to start) |
| H | `run_controls.py` | Wrong-CV WT-MetaD (two intra-CORE distances) |
| I | `run_bioemu.py` | BioEmu sequence-only models + 6 × 50 ns MD |

## Reproduce analysis (no GPU required)

```bash
conda create -n adk-analysis python=3.11 numpy scipy matplotlib mdtraj -c conda-forge
conda activate adk-analysis
python analyze_sampling.py          # needs production XTCs in output*/
python make_paper_figures.py        # rebuilds main-text and SI figures from analysis/canvas_data.json
```

`analysis/summary.json`, `analysis/method_metrics.csv`, and `analysis/figures/` are the numbers and plots used in the manuscript.

## Run the simulations (GPU)

Requires OpenMM with an OpenCL or CUDA platform, PDBFixer, and (for method I) a separate BioEmu environment.

```bash
conda activate openmm   # environment with openmm, pdbfixer, numpy
python -u run_baseline.py              # A, writes output/
python -u run_others.py                # B, C, D after A
python -u run_controls.py --nowait     # E, F, G, H
BIOEMU_PYTHON=/path/to/bioemu/python python -u run_bioemu.py --nowait
```

Typical throughput on an RTX 4090 (OpenCL) was 430–535 ns/day.

## How to cite

If you use these scripts or the AdK comparison numbers, please cite the article (Zhao, Cui, and Li) and this repository: https://github.com/liying0128/adk-conformational-sampling

## License

MIT (code). PDB coordinates remain subject to the wwPDB license. BioEmu weights are obtained from the [microsoft/bioemu](https://github.com/microsoft/bioemu) project and are not redistributed here.

## Funding

Teaching Reform Project of Beijing Union University (JJ2025Y068).
