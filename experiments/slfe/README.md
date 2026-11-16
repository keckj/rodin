# SLFE GPU research experiment

## Daytime checkpoint: nonseparable flow and FE coupling with memory

The daytime study extends the auxiliary-space transport to a prescribed
nonseparable cellular velocity. It compares oriented DG carriers, nested DG
coordinate remaps, and a unitary Fourier carrier connected to the FE space
through static FE–DG–Fourier projections. Exact polynomial shear blocks are
generated with Symox and evaluated by CUDA kernels.

The principal new construction retains the auxiliary state between steps.
For FE coefficients `v` and auxiliary values `w`, an advection step is
`w_new = S(w); v_new = v + R*(w_new - w)`. A physical FE update is applied as
`v_new = G(v); w_new = w + R(v_new - v)`. This preserves the unresolved
auxiliary memory. An isometric lifting proves an energy bound for the
composition when both sub-operators are contractive. The implementation in
`memory_transport.py` requires no projection-defect square root or inverse.

The [daytime report](report_day/rapport_journee.pdf) and
[mathematical note](report_day/memory_theory.pdf) explain the construction,
conditional consistency estimates, numerical evidence and open questions.
The study includes a spatially varying FE reaction, independent GPU matrix
exponentials for the coupling validation, and a Fourier/RK4 GPU comparator.
Both timing comparisons use CUDA graph replay. The Fourier comparator is
competitive on this smooth periodic problem; general superiority is not
established. The known ingredients and outstanding novelty assessment are
identified in the mathematical note.

The direct CUDA DG comparator uses the same triangular space, central or
upwind fluxes, and global RK4 stepping. On a strongly graded P4 mesh, the
auxiliary evolution meets an L2 tolerance of 1e-6 in a median 15.4 ms versus
482 ms for tuned upwind DG (five repeats, one field). Static setup costs
320 ms versus 14 ms, so the first complete solve is only about 1.48 times
faster. Sixteen duplicated fields measure throughput, not solution diversity.
Local-time DG has not been compared, and unresolved auxiliary scales remain
a limitation. The reaction comparison includes a uniform-mesh case where DG
is better. Raw data and failed exploratory criteria are retained separately.

LEoPart already feeds FE increments back to particle properties; that general
pattern is not claimed as new. The research question concerns the specific
contractive coupling, consistency bounds and measured costs.

Further checks extend the Fourier flow to six arbitrary integer wave
directions and a mean velocity (`fourier_modes.py`). A unimodular integer
change of coordinates turns each mode into a shear; the continuous splitting
principle has Quispel–McLaren antecedents (2003). Refinement from 95 to 767
points per direction resolves the stronger deformation in the new test.

The narrow-feature study is a counterexample to general superiority. A
graded FE mesh resolves a bump of width 0.0015 that a 95² auxiliary space
misses: L2 errors are 0.000617 for DG and 1.28 for the auxiliary scheme.
Refining both the intermediate DG space and Fourier support recovers
accuracy, at substantial cost. The mathematical note includes a lower bound
showing why time refinement alone cannot repair insufficient auxiliary
resolution.

Cooperative cross-mass assembly reduces the measured P4 block construction
from 297 to 98 ms; the earlier timing tables are retained with their original
setup costs. Large-grid tests also motivated independent cuFFT-plan ownership
for each CUDA graph. Focused memory checks and graph lifetime tests pass;
the complete racecheck log retains warnings in CuPy's nonzero scan.

An alternative direct DG construction is recorded in `rotation_dg.py` and
[its mathematical note](report_day/rotation_theory.pdf). Rank-two rotations
correct local orthogonal substeps to preserve constants and mass. Dense GPU
checks confirm order four and the invariants, but the advection tests retain
a severe accuracy restriction on the timestep. This is a documented negative
result for the intended large-step acceleration, not the selected method.

```sh
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/day_study.py --suite validation --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/fourier_study.py --suite validation --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/fourier_study.py --suite compressed --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/spectral_reference.py --graph --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/memory_study.py --output /tmp/slfe-day/memory-validation.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/reaction_study.py --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/validate_day.py --output /tmp/slfe-day/smoke.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/validate_comparisons.py --output /tmp/slfe-day/comparison-validation.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/dg_study.py --suite upwind --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/graded_study.py --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/coupled_benchmark.py --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/carrier_build_study.py --output /tmp/slfe-day/carrier-build.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/mode_study.py --suite resolution --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/scale_study.py --suite limits --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/scale_study.py --suite enrichment --output /tmp/slfe-day
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/validate_graph_lifetime.py --output /tmp/slfe-day/graph-large-validation.json
```

Recorded data are in `results_day/`; all simulation kernels run on CUDA.
The geometry is periodic and affine, and the prescribed flows have four or six
Hamiltonian modes. Arbitrary FE velocities, boundaries, local physical flux
conservation and distributed execution remain unvalidated. This is a research
checkpoint; the production Rodin variational API is unchanged.

## Follow-up: incremental auxiliary-space transport

The 16 November follow-up introduces a different linear construction:

`T = I + R* (S - I) R`.

`R` is the static exact L2 projection from triangular DG Pp to Cartesian DG
Qp, `R*` its mass adjoint, and `S` a conservative, contractive Cartesian DG
transport. Returning the increment gives mass conservation, L2 contraction
and the identity at zero displacement. The proof and its hypotheses are in
[`incremental_theory.tex`](incremental_theory.tex), with the
[follow-up PDF](report_incremental/rapport_increments.pdf) and
[compiled proof](report_incremental/incremental_theory.pdf).

`carrier.py` and `carrier.cu` implement static cross-mass construction and
matrix-free applications on CUDA. `carrier_study.py` records independent
validation, spatial refinement, small-step, long-time and resolution studies.
All 73 primary follow-up cases completed; 48 full-transfer SVD checks and a
Compute Sanitizer run passed. Data are in `results_incremental/`.

```python
from gpu import Mesh, Space, smooth
from carrier import Carrier

mesh = Mesh(32)
space = Space(mesh, 3)
carrier = Carrier(space, n=2*mesh.n)
transport = carrier.transport(1/16, flow="shear")
coefficients = space.project(smooth)
coefficients = transport.apply(coefficients)
```

The initial carrier supports translation and horizontal incompressible shear.
It is not a general velocity FE solver. At equal grid resolution and degree,
the carrier has fewer degrees of freedom than the triangular space and leaves
some modes stationary; the finer carrier in this example removes the nullspace
in the small systems checked by SVD. Accuracy, carrier resolution and general
flow extensions remain research questions. Increment transfer has FLIP
antecedents; novelty is not asserted for that principle.

```sh
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/carrier_study.py --suite validation --output /tmp/slfe-incremental
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/carrier_study.py --suite convergence --output /tmp/slfe-incremental
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/carrier_study.py --suite small_steps --output /tmp/slfe-incremental
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/carrier_study.py --suite long_time --output /tmp/slfe-incremental
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/carrier_study.py --suite resolution --output /tmp/slfe-incremental
```

Other follow-up experiments are retained separately. `normalized.py` implements
interface-layer quadrature and a local Gram correction: the latter failed
spatial consistency checks and is not the selected method. `flux.py` and
`ray_flux.cu` provide a characteristic face-flux comparison with an optional
nonlinear energy constraint. Their exploratory JSONL files are distinct from
the reproducible `carrier-*` study records and are not counted as validations
of the selected incremental algorithm.

## Original overnight campaign

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
