"""GPU translation consistency under frequent prescribed support changes.

Distributed under the Boost Software License, Version 1.0.
Two Fourier pairs give an exact FE reference by component rotation. The
Frobenius norm of the local consistency defect is invariant under this
rotation, so it is measured once per support, not estimated from the run.
The reference-relative stability theorem predicts the sum of these defects
as an upper bound for the final FE error relative to its exact projection.
All projection, evolution, reference rotations and reductions execute on GPU.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_projection import TensorProjection
from memory_transport import MemoryTransport
from graph_memory import GraphMemory
from residual_energy import ResidualEnergy
from campaign import provenance


def profile(points, time):
    x, y = points[..., 0] - time, points[..., 1] - 0.37 * time
    a, b = 2 * cp.pi * x, 2 * cp.pi * (x + 2 * y)
    return cp.stack((cp.cos(a), cp.sin(a), cp.cos(b), cp.sin(b)), axis=-1)


def reference(initial, time):
    angle = 2 * cp.pi * cp.asarray((1.0, 1.74)) * time
    cosine, sine = cp.cos(angle), cp.sin(angle)
    result = cp.empty_like(initial)
    for pair in range(2):
        a, b = initial[:, 2 * pair], initial[:, 2 * pair + 1]
        result[:, 2 * pair] = cosine[pair] * a + sine[pair] * b
        result[:, 2 * pair + 1] = cosine[pair] * b - sine[pair] * a
    return result


def evolve(space, carriers, crosses, initial, steps, method, remap_every=1):
    duration = 0.2
    dt = duration / steps
    flows = [c.transport(dt) for c in carriers]
    create = GraphMemory if method == "graph" else MemoryTransport
    next_reference = reference(initial, dt)
    defects, minimum_defect_squares = [], []
    for c, flow in zip(carriers, flows):
        if method == "graph":
            old, new = create(c, initial), create(c, next_reference)
            squared = c.energy(flow.apply_carrier(old.auxiliary) - new.auxiliary)
            squared += space.energy(old.hidden_fe - new.hidden_fe)
        else:
            difference = initial - next_reference
            squared = c.energy(
                flow.apply_carrier(c.project(initial)) - c.project(next_reference)
            )
            squared += ResidualEnergy(c).apply(difference, c.project(difference))
        # Positive common quadrature avoids cancellation of two integrated norms.
        minimum_defect_squares.append(float(cp.min(squared)))
        assert minimum_defect_squares[-1] > -1e-18
        defects.append(float(cp.sqrt(cp.sum(cp.maximum(squared, 0)))))
    state = create(carriers[0], initial)
    initial_energy, initial_mass = state.energy(), space.mass(initial)
    energy_ratio, mass_error, observation_error = 1.0, 0.0, 0.0
    support_steps = [0, 0]
    current = 0
    cp.cuda.Stream.null.synchronize()
    begin = time.perf_counter()
    for step in range(steps):
        state.advect(flows[current].apply_carrier)
        support_steps[current] += 1
        if (step + 1) % remap_every == 0 and step + 1 < steps:
            target = 1 - current
            if method == "graph":
                state.transfer(carriers[target], crosses[current].apply_carrier)
            elif method == "galerkin":
                state.project_memory_galerkin(
                    carriers[target],
                    crosses[current].apply_carrier,
                    iterations=2,
                    shared=True,
                    gram_backend="reduction",
                )
            else:
                state.restart(carriers[target])
            current = target
        if (step + 1) % 16 == 0 or step + 1 == steps:
            energy_ratio = max(
                energy_ratio, float(cp.max(state.energy() / initial_energy))
            )
            mass_error = max(
                mass_error,
                float(cp.max(cp.abs(space.mass(state.value) - initial_mass))),
            )
            observed = (
                state.observe()
                if method == "graph"
                else state.anchor
                + state.coupling.adjoint(
                    state.auxiliary - state.coupling.project(state.anchor)
                )
            )
            observation_error = max(
                observation_error,
                float(cp.sqrt(cp.sum(space.energy(state.value - observed)))),
            )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - begin
    exact_fe = reference(initial, duration)
    bound = float(cp.dot(cp.asarray(support_steps), cp.asarray(defects)))
    error = float(cp.sqrt(cp.sum(space.energy(state.value - exact_fe))))
    assert error <= bound + 1e-10, (error, bound)
    assert energy_ratio <= 1 + 1e-9 and mass_error < 1e-9, (energy_ratio, mass_error)
    errors = space.errors(state.value, profile, duration, degree=20, subdivisions=2)
    floor = space.errors(exact_fe, profile, duration, degree=20, subdivisions=2)
    return dict(
        n=space.mesh.n,
        degree=space.degree,
        steps=steps,
        remap_every=remap_every,
        changes=(steps - 1) // remap_every,
        method=method,
        duration=duration,
        local_defects=defects,
        minimum_defect_squares=minimum_defect_squares,
        defect_energy=(
            "positive common quadrature"
            if method != "graph"
            else "positive graph coordinates"
        ),
        summed_defect_bound=bound,
        fe_reference_frobenius_error=error,
        bound_ratio=error / bound,
        energy_ratio=energy_ratio,
        mass_error=mass_error,
        observation_error=observation_error,
        evolution_seconds=elapsed,
        best_fe_l2=floor["l2"],
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", nargs="+", type=int, default=[4])
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--remap-every", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("reset", "galerkin", "graph"),
        default=["reset", "galerkin", "graph"],
    )
    args = parser.parse_args()
    if args.remap_every < 1 or min(args.steps) < 1:
        raise ValueError("Positive step counts and remapping intervals are required")
    meta = provenance()
    with args.output.open("w") as stream:
        for degree in args.degrees:
            for n in args.sizes:
                space = Space(Mesh(n, jitter=0.1, seed=918), degree)
                initial = space.project(profile, degree=20)
                line = cp.linspace(0, 1, n + 1)
                displaced = line + 0.24 / n * cp.sin(2 * cp.pi * line)
                grids = [
                    cp.stack((line, line)),
                    cp.stack((displaced, 1 - displaced[::-1])),
                ]
                carriers = [
                    Carrier(space, n=n, edges=edges, builder="tiled") for edges in grids
                ]
                crosses = [
                    TensorProjection(carriers[i], carriers[1 - i]) for i in range(2)
                ]
                for steps in args.steps:
                    for method in args.methods:
                        print(
                            json.dumps(
                                dict(n=n, degree=degree, steps=steps, method=method)
                            ),
                            flush=True,
                        )
                        result = evolve(
                            space,
                            carriers,
                            crosses,
                            initial,
                            steps,
                            method,
                            args.remap_every,
                        )
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
