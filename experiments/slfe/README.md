# SLFE GPU research experiment

## Night of November 16–17: moving supports and memory projection

The [illustrated night report](report_night2/rapport_nuit.pdf),
[HTML version](report_night2/rapport_nuit.html), and
[mathematical note](report_night2/restart_theory.pdf) cover variable auxiliary
spaces. Actual FE fields, error maps, convergence curves, complete costs and
failed adaptive refinement experiments are included. All numerical work uses
CUDA; host code orchestrates experiments and renders figures.

`MemoryTransport` now retains an FE anchor `a`, so that
`v = R*w + (I-R*R)a`. This permits a contractive restart, a partial memory
transfer, a lossless nested refinement, and an iterative projection onto the
sum of the FE and new auxiliary spaces. The last method preserves the FE
observation at every finite iteration and requires exact projections in a
common ambient L2 space, not merely arbitrary contractive maps. Its stationary
iteration is classical linear algebra; novelty of the complete construction
is not established. Dense GPU oracles reconstruct independent lifted and
common-ambient representations. Focused Compute Sanitizer checks pass.

For a moving localized structure on 32768 graded P4 triangles, the pure
advection baseline with a 64×64 Q4 moving carrier meets L2 < 1e-3 for every
one of 1, 4 and 16 distinct fields. Three interleaved repetitions measure
complete-cost ratios of 8.84, 13.37 and 34.67 against global-step upwind
DG/RK4. This comparison includes setup and endpoint transfers, and excludes
common mesh construction, initial FE projection and reference diagnostics.
It does not compare local-time DG or an external GPU SL-DG solver. The pure
advection baseline does not provide intermediate FE physics operations.

For the coupled method, preserving more memory reduces the observed error
after six support changes. With spatial FE reaction, 128 steps and a 64²
FE mesh, eight projection iterations give L2 = 0.002728 versus 0.004535 for
a reset; the FE projection floor is 0.002720. The nested-refinement identity
passes its Gram tests, but its initial controllers exhaust their cell budget
early and fail the transport accuracy target. These failures are retained.

Subsequent experiments add a contractive squared-Chebyshev memory filter,
2000-change invariant tests, and energy-based partitions in all four shear
directions. For four separated structures, the latter reduce the maximum
pure-advection error from 0.07135 to 0.005902 at auxiliary n=64, against an
FE floor of 0.005891. Coupled reaction and finer-FE tests are recorded
separately. Deform-and-return experiments expose sensitivity to regridding
frequency; this controller is still experimental.

`hamiltonian_commutator.py` explores prescribed streamfunctions expressed as
sums of products a(y)b(x). It uses the published PCP16 commutator formula
of Casas, Escorihuela-Tomas and Moreno Casares (2025), with full attribution.
Dense GPU matrix oracles verify the local third-order defect in physical
time, unitarity and inverse. A rank-one vortex reaches nodal RMS error
3.262e-5 at 256 steps on 255² points, unchanged on 511² points. These nodal
results are not substituted for FE quadrature errors or a cost comparison
against a tuned spectral solver. The subsequent campaign tests a GPU-SVD
input path and further FE cases; general FE velocities remain outside the
validated scope.

The subsequent SVD experiment is a negative performance result: on the
nonseparable vortex at 255² points, Fourier/RK4 reaches nodal RMS 5.28e-7 in
0.143 s including setup; the rank-11 PCP16 composition reaches 7.66e-6 in
5.07 s. Its 336 shears per step are not competitive on this problem.

A separate consistency defect was found in finite cold-start memory
projection: it damps hidden state even when the support does not change.
`warm_validation.py` reproduces this defect and tests a warm initialization
with a radial energy limiter. This variant preserves the ambient state for
an unchanged support and bounds its energy, but is nonlinear; no Lipschitz
stability or systematic accuracy improvement follows. `warm_study.py`
separates it from changes in physical adaptation frequency. With 512 steps,
holding the check interval equal to that of 128 steps reduces return error
from 0.00127 to 0.000870 for the cold transfer. Oversampling, monitor floors
and FE-based monitors are tested separately in `monitor_study.py`.

`compressible_study.py` extends the forward weak deposition to a separable
compressible characteristic map, using target-edge preimages to split the
source integration. Its exact reference includes the inverse Jacobian.
Mass conservation and physical L2 growth are distinct checks. The ordinary
GPU sparse path is used because CuPy 13's layout conversion inside the CSR
product does not support CUDA graph capture.

`night_queue.py` runs the remaining GPU campaigns sequentially and records
exit codes in `queue-status.json`. The long return campaign supports
`--resume`; completed records survive a change in research priorities.

Material energy samples reproduce the large periodic-grid movements without
any FE/DG field evolution. Shifting the periodic cut changes these movements;
the first floating-phase policy minimizes cumulative-coordinate displacement
but does not improve the maximum-edge-motion trigger or return accuracy.
`material_monitor.py` and `phase_study.py` retain these controls.

A further experimental transfer, `project_memory_galerkin`, projects hidden
memory onto a small Krylov space augmented by the old transferred state.
In exact arithmetic it preserves the FE observation and satisfies an exact
Pythagorean remapping identity. The GPU implementation uses a small Gram
pseudoinverse per field, with a numerical rank threshold. Its dedicated
validation passes independent ambient-space oracles for degrees 1, 2 and 4,
including the Pythagorean identity and monotonic energy retention as directions
are added. On the four-field return test with 512 steps and checks every four
steps, its maximum L2 error is 0.000742 versus 0.001270 for eight cold iterations.
This is not an established cost improvement: a shorter Chebyshev run reaches
0.000778 in 17.2 s, versus 77.9 s for that Galerkin run. The unchanged-support
identity addresses a distinct consistency defect.

Further controls separate the transfer from the carrier resolution and from
physical FE operations. A minimax periodic phase reduces material-monitor
changes from 32 to 23, but does not systematically improve return accuracy.
Pure auxiliary return tests show strong dependence on auxiliary resolution:
at duration one per leg and 512 steps, increasing n from 128 to 256 reduces
the maximum error from 0.0232 to 0.00270. These tests omit intermediate FE
operations and cannot establish the accuracy of the coupled method.

`fe_diffusion.py` implements matrix-free SIPG diffusion and Crank--Nicolson
with preconditioned CG on CUDA. Dense GPU matrix and eigenvalue controls,
manufactured heat convergence and translated Gaussian tests check the
implementation. The coupled diffusion campaign retains 20 cases; preserving
memory gives only a modest benefit in some regimes. `physics_fields.py`
provides independently checked field and error maps for diffusion and sources.
Source-triggered nested refinement preserves the ambient state, but cannot
undo errors accumulated before refinement.

`remap_controller.py` uses the Galerkin Pythagorean identity to budget actual
compression distances. Uniformly dividing the budget among future checks
severely restricts mesh motion and worsens transport accuracy in the tested
cases. The bound concerns compression along the realized evolution operators;
it does not bound their PDE approximation error. A remaining-budget policy is
tested separately. `frozen_transfer_study.py` compares transfers on a shared
material-driven geometry schedule. `krylov_study.py` adds a classical GPU
Arnoldi exponential-DG competitor, with an independent dense-matrix oracle.
Results of these further campaigns are included only after execution.

The identical-geometry comparison isolates an implementation bottleneck in
the augmented transfer: tiny Gram outputs with very long reduction dimensions
are expensive through the original CuPy contractions. `weighted_products.py`
uses a fused CUDA reduction over spatial chunks. On the four-field P4 test,
three-direction transfers take 8.08 ms instead of 126.93 ms (five interleaved
repeats); the retained ranks agree and the ambient state difference is below
2e-11 across the tested configurations. Near-null directions amplify anchor
coefficient differences, which are recorded separately. Independent weighted
products, degree-1/2/4 transfer comparisons and Compute Sanitizer pass.
The `gram_backend="reduction"` option is explicit; historical timings retain
their original backend. Full-solve savings remain limited by geometric rebuilds.
An alternative tiny Jacobi eigensolver is retained as a diagnostic experiment;
the eigensolve was not the bottleneck and that alternative is not selected.

`shared=True` forms one augmented space from all fields. Its common projection
preserves orthogonal changes of component coordinates and makes the full
cross-field Gram loss positive semidefinite. The 48 independent controls in
`block_validation.py` check these properties and the ambient Pythagorean
identity. This is a block-projection construction, not a claim of a new Krylov
method. It remains nonlinear in the ensemble of states. In particular,
`galerkin_stability.py` gives explicit scalar-transfer counterexamples with
pairwise distance gains approaching 100 and 890 despite energy contraction.

The exponential-DG competitor now includes smaller time steps that meet the
same physical tolerance. Its initial upwind run exposed overflow in CuPy's
intermediate trace shift for stiff dissipative matrices; external scaling and
GPU squaring resolve this on an independent spectral oracle. Failed accuracy
settings and the successful refinements remain in the report. The subsequent
system-reaction study compares the same coupled FE problem in two component
bases, using a characteristic reference and a common material support schedule.

```sh
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/ambient_validation.py --output /tmp/slfe-ambient.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/adaptive_study.py --suite physics --output /tmp/slfe-physics.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/matched_benchmark.py --output /tmp/slfe-matched.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/diffusion_study.py --output /tmp/slfe-diffusion.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/frozen_transfer_study.py --output /tmp/slfe-frozen.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/krylov_study.py --output /tmp/slfe-krylov.jsonl
python3 experiments/slfe/night_finalize.py --input /home/keckj/Documents/slfe-results-night2
```

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
The opening figures show the reconstructed FE fields, independent
characteristic reference, pointwise errors, local mesh resolution and costs
for a smooth profile and the unresolved narrow-feature counterexample.
The [HTML report](report_day/rapport_journee.html) provides the same figures.
Numerical field evaluation runs in `field_sample.cu`; the renderer preserves
cell-local DG values and uses common color scales across methods. Element
interior evaluations are checked against direct basis evaluation, independently
of the locator. The focused Compute Sanitizer log is retained in
`results_day/field-sampler-sanitizer.log.txt`.

```sh
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/field_report.py --output /home/keckj/Documents/slfe-results-day/visuals
python3 experiments/slfe/day_finalize.py --input /home/keckj/Documents/slfe-results-day
```

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

The subsequent graded-carrier prototype resolves this specific counterexample.
`Carrier(..., edges=...)` accepts nonuniform periodic tensor partitions.
`tensor_remap.py` integrates each one-dimensional polynomial overlap exactly
on CUDA, while `tensor_cellular.py` connects differently graded coordinate
supports through mass-adjoint row projections. The principal state persists
between steps. With 128² Q4 cells and four steps, the narrow cellular test
has L2 error 0.000480, close to the FE projection error 0.000479. This requires
409600 principal auxiliary values and two oriented supports, and does not
establish a general cost advantage over the 9025-value Fourier construction.
The partitions are fixed around the known initial feature, not adapted in time.
The [graded-support note](report_day/tensor_theory.pdf) states the discrete
contraction proof and the transverse collocation approximation.

```sh
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/validate_tensor.py --output /tmp/slfe-tensor-validation.json
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/tensor_study.py --suite resolution --output /home/keckj/Documents/slfe-results-day/tensor-resolution.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/tensor_study.py --suite cellular --output /home/keckj/Documents/slfe-results-day/tensor-cellular.jsonl
CUDA_PATH=/usr/local/cuda python3 experiments/slfe/tensor_report.py --input /home/keckj/Documents/slfe-results-day
```

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

The follow-up ensemble campaign runs 60 combinations of mesh seed, resolution
and number of distinct Gaussian profiles. Each isolated GPU worker tunes the
two methods against a common all-field L2 tolerance, retaining rejected
settings and unresolved cases. Three timing repeats are recorded. The user
service `slfe-day-campaign-20261116` refreshes and publishes the report when
this finite campaign completes, with a deadline of 17:40 Europe/Paris.
`slfe-day-report-20261116.timer` refreshes it again at 18:00 on 16 November.
These services execute predefined numerical tests and report generation;
they do not perform further interactive algorithm design.

```sh
systemctl --user status slfe-day-campaign-20261116.service
systemctl --user list-timers slfe-day-report-20261116.timer
cat /home/keckj/Documents/slfe-results-day/ensemble-status.json
```

`day_finalize.py` refuses another branch or push destination, and refuses to
commit when the index already has staged changes. It stages only generated
daytime artifacts. The sole publication destination remains `keckj/rodin:SLFE`.

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
