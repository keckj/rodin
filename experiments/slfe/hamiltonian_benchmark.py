"""CUDA-graph comparisons on identical FE spaces and filtered velocities.

Distributed under the Boost Software License, Version 1.0.
Both SLFE methods update the FE observation at every captured step. DG/RK4
uses several explicitly recorded step sizes; unstable random-state probes
are rejected before timing. These probes are empirical stability controls,
not an unconditional RK4 theorem. All references and reductions use CUDA.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from gpu import cp, HERE, Mesh, Space, launch
from carrier import Carrier
from tensor_hamiltonian import (
    TensorHamiltonian,
    ComposedHamiltonian,
    streamfunction_modes,
)
from hamiltonian_study import WAVES, profile, velocity
from eulerian_dg import EulerianDG
from gpu_graph import EvolutionGraph
from krylov_study import ErrorOracle
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--sl-steps", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--scales", nargs="+", type=float, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--central-scales", nargs="+", type=float)
    parser.add_argument("--upwind-scales", nargs="+", type=float)
    parser.add_argument(
        "--scheme",
        choices=("strang", "triple4", "suzuki4", "yoshida6"),
        default="strang",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("dg-central", "dg-upwind", "memory", "reset"),
        default=["dg-central", "dg-upwind", "memory", "reset"],
    )
    parser.add_argument("--reverse-case-order", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    end = 0.5
    waves = cp.asarray(WAVES, dtype=cp.float64)
    denominator = cp.sum(waves**2, axis=1)
    analytic = cp.stack(
        (0.018 / denominator, 0.009 * (-1.0) ** cp.arange(len(WAVES)) / denominator),
        axis=1,
    )
    modes = cp.ascontiguousarray(cp.concatenate((waves, analytic), axis=1))
    reference = cp.RawKernel(
        (HERE / "hamiltonian_reference.cu").read_text(), "hamiltonian_reference"
    )
    reference.compile()

    def exact(points):
        flat = cp.ascontiguousarray(points.reshape(-1, 2))
        feet = cp.empty_like(flat)
        launch(
            reference, len(flat), (flat, feet, len(flat), modes, len(WAVES), -end, 512)
        )
        return profile(feet.reshape(points.shape), 0)

    def psi(points, t):
        phase = 2 * cp.pi * cp.einsum("...i,mi->...m", points, waves)
        return cp.sum(
            cp.cos(phase) * analytic[:, 0] + cp.sin(phase) * analytic[:, 1], axis=-1
        )[..., None]

    with args.output.open("w") as output:
        for n in args.sizes:
            space = Space(Mesh(n, jitter=0.18, seed=421), 4)
            initial = space.project(profile, degree=20)
            coefficients = streamfunction_modes(
                space, space.project(psi, degree=20), WAVES
            )
            oracle = ErrorOracle(space, profile, end, reference=exact)
            norm = cp.sqrt(
                cp.sum(oracle.reference**2 * oracle.weights[..., None], axis=(0, 1))
            )
            floor = (cp.asarray(oracle.errors(oracle.best)["l2"]) / norm).tolist()
            carrier = Carrier(space, n=2 * n, builder="tiled")
            initial_w = carrier.project(initial)
            wshape = initial_w.shape
            wsize = initial_w.size
            packed_initial = cp.concatenate((initial_w.ravel(), initial.ravel()))
            probe = cp.random.RandomState(631).normal(size=initial.shape)
            packed_probe = cp.concatenate(
                (carrier.project(probe).ravel(), probe.ravel())
            )
            initial_mass = space.mass(initial)
            cases = [
                (method, scale)
                for method in args.methods
                if method.startswith("dg-")
                for scale in (
                    (
                        args.central_scales
                        if method == "dg-central"
                        else args.upwind_scales
                    )
                    or args.scales
                )
            ]
            cases += [
                (method, steps)
                for method in args.methods
                if not method.startswith("dg-")
                for steps in args.sl_steps
            ]
            if args.reverse_case_order:
                cases.reverse()
            for method, parameter in cases:
                cp.cuda.Stream.null.synchronize()
                begin = time.perf_counter()
                if method.startswith("dg-"):
                    dg = EulerianDG(
                        space,
                        flux=method[3:],
                        velocity_field=lambda p, t: velocity(p, coefficients),
                    )
                    steps = max(1, math.ceil(end * dg.norm_bound / (2 * parameter)))
                    graph = EvolutionGraph(dg.step, initial, end / steps)
                    state0, random0 = initial, probe

                    def diagnostic(value):
                        return space.energy(value)

                    skew = dg.skew_correction
                else:
                    steps = int(parameter)
                    flow = (
                        TensorHamiltonian(carrier, WAVES, coefficients, end / steps)
                        if args.scheme == "strang"
                        else ComposedHamiltonian(
                            carrier, WAVES, coefficients, end / steps, args.scheme
                        )
                    )

                    def operation(value, dt):
                        w = value[:wsize].reshape(wshape)
                        v = value[wsize:].reshape(initial.shape)
                        new = flow.apply_carrier(w)
                        observation = v + carrier.adjoint(new - w)
                        if method == "reset":
                            new = carrier.project(observation)
                        return cp.concatenate((new.ravel(), observation.ravel()))

                    graph = EvolutionGraph(operation, packed_initial, end / steps)
                    state0, random0 = packed_initial, packed_probe

                    def diagnostic(value):
                        w = value[:wsize].reshape(wshape)
                        v = value[wsize:].reshape(initial.shape)
                        anchor = v if method == "reset" else probe
                        return (
                            carrier.energy(w)
                            + space.energy(anchor)
                            - carrier.energy(carrier.project(anchor))
                        )

                    skew = None
                cp.cuda.Stream.null.synchronize()
                setup = time.perf_counter() - begin
                # Reconstruct the same captured operator in the ordinary path.
                if not method.startswith("dg-"):
                    expected = operation(state0, cp.asarray(end / steps))
                    graph.reset(state0)
                    graph.launch(end / steps)
                    capture_error = float(cp.max(cp.abs(graph.result() - expected)))
                    assert capture_error < 1e-10, capture_error
                else:
                    capture_error = None
                graph.reset(random0)
                energy0 = space.energy(probe)
                stable = True
                maximum = 1.0
                for step in range(steps):
                    graph.launch(end / steps)
                    if (step + 1) % 32 == 0 or step + 1 == steps:
                        ratio = float(cp.max(diagnostic(graph.result()) / energy0))
                        maximum = max(maximum, ratio)
                        if not math.isfinite(ratio) or ratio > 1.001:
                            stable = False
                            break
                result = dict(
                    nfe=n,
                    naux=2 * n,
                    fields=2,
                    degree=4,
                    end=end,
                    method=method,
                    scheme="rk4" if method.startswith("dg-") else args.scheme,
                    parameter=parameter,
                    steps=steps,
                    stable_probe=stable,
                    probe_max_energy_ratio=maximum,
                    capture_error=capture_error,
                    skew_correction=skew,
                    setup_seconds=setup,
                    relative_fe_floor=floor,
                )
                if stable:
                    timings = []
                    for repeat in range(args.repeats):
                        graph.reset(state0)
                        graph.stream.synchronize()
                        begin = time.perf_counter()
                        for _ in range(steps):
                            graph.launch(end / steps)
                        final = graph.result()
                        timings.append(time.perf_counter() - begin)
                    value = (
                        final
                        if method.startswith("dg-")
                        else final[wsize:].reshape(initial.shape)
                    )
                    errors = oracle.errors(value)
                    result.update(
                        evolution_seconds=statistics.median(timings),
                        repetitions_seconds=timings,
                        relative_l2=(cp.asarray(errors["l2"]) / norm).tolist(),
                        mass_error=float(
                            cp.max(cp.abs(space.mass(value) - initial_mass))
                        ),
                        **errors
                    )
                    assert result["mass_error"] < 1e-9, result["mass_error"]
                output.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
                output.flush()
                print(json.dumps(result), flush=True)
                del graph
                cp.get_default_memory_pool().free_all_blocks()
            del oracle
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
