# SLFE GPU research experiment

Branch `SLFE`, based on `838bc93ae71`. Research deadline: 16 November 2026,
08:00 Europe/Paris. Only `https://github.com/keckj/rodin.git` is an authorized
publication destination. The original worktree and HySoP remain untouched.

This directory contains an experimental CUDA implementation driven by CuPy.
It is not yet a Rodin variational operator. Mesh generation, trajectory maps,
point location, intersections, transfer, field evolution and error reductions
execute on the GPU. Python orchestrates launches and records results. Symox
generates and checks mathematical constants before the runs. Plotting and
report preparation are host tasks.

## Questions

1. Can quadrature-particle deposition onto DG triangles simultaneously retain
   mass, constants, accuracy and an acceptable amplification factor?
2. What is the GPU cost of an accurately integrated geometric transfer, at
   the same field error and on the same mesh?
3. Can moment-preserving kernels from the HySoP construction suggest a less
   expensive transfer? Cartesian performance is a separate reference, not
   evidence of unstructured-mesh performance.

## Scope and planned experiments

Start with periodic affine triangles and exact translation, using nested
mean-orthonormal DG P0 through P3 bases. Jittered vertices and varied diagonals
provide nonuniform triangle shapes. The first locator exploits a bounded
perturbation of a Cartesian topology and is not a general unstructured BVH.

Compare source quadrature deposition with polygon-intersection integration.
Test zero motion, constants, random fields, mass, energy amplification,
space/time refinement, repeated translation, shear and reversible deformation.
Record setup and application separately, including multi-field amortization.
Investigate failed invariants before interpreting timing comparisons.

The broad SL-FE idea is established. Originality and publication quality are
research questions, not properties asserted by this prototype.

## Provenance

Triangle quadrature is read from Rodin's existing Xiao–Gimbutas tables, retaining
their attribution in `src/Rodin/QF/XiaoGimbutasData.h`. The orthonormal basis is
derived with Symox from exact simplex moments. The lambda kernel constraints
are those of HySoP's `hysop/numerics/remesh/kernel_generator.py` and
`remesh.py`, inspected at HEAD `6d62e61763efdd891546ed89904efa05be781f96`.
No runtime HySoP dependency or copied implementation is introduced.

## Execution

```sh
python3 experiments/slfe/symbolic.py
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/validate.py --output /tmp/slfe-validation.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/validate_moments.py --output /tmp/slfe-moments.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/campaign.py --suite translation --output /tmp/slfe-results
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/research.py --suite moments --output /tmp/slfe-results
python3 experiments/slfe/report.py --input /tmp/slfe-results --output /tmp/slfe-report
```

CuPy 13.6, CUDA 12.9 and an NVIDIA RTX 5090 were detected at campaign start.
There is no CPU numerical fallback. Runtime outputs are written to the
explicit campaign directory; the report records actual commands and versions.

## Compatible cubature candidate

`moments.py` fits a common positive cubature rule to source and target moments
through degree `2p`. `theory.tex` proves conditional mass conservation,
constant preservation and L2 contraction, and gives the compressible extension.
It also proves a small-displacement obstruction: an isolated cell cannot
satisfy both sets of first moments for a nonzero translation, even with signed
weights. Fixed interior particle supports therefore cannot handle all time
steps. The active-set prototype is not a complete feasibility solver. Failures,
unconverged constraints and negative weights remain in the recorded results.

The `support` suite probes this obstruction and the `robustness` suite varies
mesh seeds, displacements and particle densities. `curved_moments` uses exact
nonlinear shear/compression trajectories. These runs are diagnostic, not a
claim of a fast general-purpose method or a demonstrated new paper result.

## Overnight delivery, 15–16 November 2026

The campaign runs as the persistent user service `slfe-campaign-20261115`.
It inhibits automatic sleep while working and stops at 07:35 Europe/Paris.
The user timer `slfe-report-20261116.timer` starts finalization at 07:50.
`finalize.py` builds PDF/HTML reports, preserves the numerical records in
`results/`, and commits/pushes generated artifacts only to `keckj/rodin:SLFE`.
It refuses another branch or another fork push URL and never forces a push.
This prepares files and publishes the fork; it does not send a chat message.

```sh
systemctl --user status slfe-campaign-20261115.service
systemctl --user list-timers slfe-report-20261116.timer
cat /home/keckj/Documents/slfe-results/overnight-status.json
cat /home/keckj/Documents/slfe-results/delivery.json
```

The local report is `/home/keckj/Documents/slfe-results/report/rapport.pdf`.
An earlier draft is retained on the fork before the scheduled final update.
The sibling worktree `/home/keckj/Documents/rodin-SLFE` contains all changes;
the original `rodin` worktree is not modified. No Rodin library module is
changed, so validation targets these CUDA experiments rather than rebuilding
the unrelated production library.
