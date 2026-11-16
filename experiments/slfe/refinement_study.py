"""Nested auxiliary refinement driven by a projected FE advection residual.

Distributed under the Boost Software License, Version 1.0.
The indicator ||A v||^2-||R A v||^2 measures unresolved FE-generator content.
It is one term of the consistency defect, not a complete PDE-error estimator.
Only refinement is implemented; spent fine intervals are not yet coarsened.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_projection import TensorProjection
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from eulerian_dg import EulerianDG
from adaptive_study import fe_center
from moving_study import profile_family
from day_study import reference_feet
from campaign import provenance


def evolve(
    nfe=128, end=0.25, steps=128, tolerance=0.1, maximum=96, count=4, strategy="detail"
):
    space = Space(GradedMesh(nfe, 0.95), 4)
    profile = profile_family(0.01, 1)
    exact = lambda points, t: profile(reference_feet(points, -end, 512), 0)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    generator = EulerianDG(space, flux="upwind")
    carrier = Carrier(space, n=16, edges=cp.stack((cp.linspace(0.0, 1.0, 17),) * 2))
    state = MemoryTransport(carrier, initial)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    history = []
    flow = None
    energy0 = space.energy(initial)
    start = time.perf_counter()
    refinement_cost = 0.0
    for first in range(0, steps, 4):
        derivative = generator.rhs(state.value)
        source_energy = space.energy(derivative)
        refinements = []
        while True:
            projected = carrier.project(derivative)
            missing_squared = source_energy - carrier.energy(projected)
            indicator = float(cp.sqrt(cp.maximum(cp.max(missing_squared), 0)))
            if indicator <= tolerance or carrier.n >= maximum:
                break
            cp.cuda.Stream.null.synchronize()
            begin = time.perf_counter()
            n, p = carrier.n, carrier.np
            if strategy == "detail":
                # The difference lives in W_fine orthogonal to W_current.
                # Select parents by the newly resolved derivative energy,
                # rather than repeatedly splitting already resolved peaks.
                midpoints = (carrier.edges[:, :-1] + carrier.edges[:, 1:]) / 2
                candidate_edges = cp.sort(
                    cp.concatenate((carrier.edges, midpoints), axis=1), axis=1
                )
                candidate = Carrier(space, n=2 * n, edges=candidate_edges)
                inject = TensorProjection(carrier, candidate)
                detail = candidate.project(derivative) - inject.apply_carrier(projected)
                energy = (cp.sum(detail * detail, axis=1) * candidate.mass).reshape(
                    2 * n, 2 * n, p, p
                )
                marginal = [
                    cp.sum(energy, axis=(0, 2, 3)).reshape(n, 2).sum(axis=1),
                    cp.sum(energy, axis=(1, 2, 3)).reshape(n, 2).sum(axis=1),
                ]
            else:
                energy = cp.sum(projected * projected, axis=1) * carrier.mass
                energy = energy.reshape(n, n, p, p)
                marginal = [
                    cp.sum(energy, axis=(0, 2, 3)),
                    cp.sum(energy, axis=(1, 2, 3)),
                ]
            added = min(count, maximum - n, n)
            edges = []
            for axis in (0, 1):
                selected = cp.argsort(marginal[axis])[-added:]
                midpoints = (carrier.edges[axis, :-1] + carrier.edges[axis, 1:]) / 2
                edges.append(
                    cp.sort(cp.concatenate((carrier.edges[axis], midpoints[selected])))
                )
            finer = Carrier(space, n=n + added, edges=cp.stack(edges))
            injection = TensorProjection(carrier, finer)
            loss = state.refine(finer, injection.apply_carrier)
            assert float(cp.max(cp.abs(loss))) < 5e-9
            carrier = finer
            flow = None
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - begin
            refinement_cost += elapsed
            refinements.append(
                dict(
                    old_n=n,
                    new_n=carrier.n,
                    indicator=indicator,
                    energy_loss=loss.tolist(),
                    seconds=elapsed,
                )
            )
        if flow is None:
            center = fe_center(space, state.value)
            flow = TensorCellular(carrier, end / steps, 0.95, center=center)
        chunk = min(4, steps - first)

        def operation(w):
            for _ in range(chunk):
                w = flow.apply_carrier(w)
            return w

        state.advect(operation)
        ratio = float(cp.max(state.energy() / energy0))
        mass_error = float(
            cp.max(cp.abs(space.mass(state.value) - space.mass(initial)))
        )
        assert ratio < 1 + 3e-9 and mass_error < 3e-9
        history.append(
            dict(
                step=first + chunk,
                n=carrier.n,
                indicator=indicator,
                tolerance_met=indicator <= tolerance,
                refinements=refinements,
                energy_ratio=ratio,
                mass_error=mass_error,
            )
        )
        print(
            json.dumps(dict(step=first + chunk, n=carrier.n, indicator=indicator)),
            flush=True,
        )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(state.value, exact, 0, degree=20, subdivisions=2)
    floor = space.errors(
        space.project(exact, degree=20), exact, 0, degree=20, subdivisions=2
    )["l2"]
    return dict(
        nfe=nfe,
        end=end,
        steps=steps,
        tolerance=tolerance,
        maximum=maximum,
        strategy=strategy,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        refinement_seconds=refinement_cost,
        final_n=carrier.n,
        best_fe_l2=floor,
        history=history,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--maximum", type=int, default=96)
    parser.add_argument("--strategy", choices=("energy", "detail"), default="detail")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evolve(
        tolerance=args.tolerance, maximum=args.maximum, strategy=args.strategy
    )
    args.output.write_text(
        json.dumps(dict(result=result, provenance=provenance()), indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}))


if __name__ == "__main__":
    main()
