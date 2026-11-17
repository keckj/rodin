"""GPU closure diagnostic: reaction creates unresolved scales from a constant.

Distributed under the Boost Software License, Version 1.0.
The exact observable is P_V exp(-T*r), not exp(-T*P_V*r|_V) applied to one.
A resolved auxiliary-only evolution is included as a direct fine-grid
baseline; this experiment tests the need for unresolved state, not novelty.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Mesh, Space, quadrature
from carrier import Carrier
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from residual_energy import ResidualEnergy
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--naux", type=int, default=128)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("full", "target", "reset", "auxiliary"),
        default=["full", "reset", "auxiliary"],
    )
    args = parser.parse_args()
    strength, amplitude = 20.0, 0.95
    records = []
    meta = provenance()
    with args.output.open("w") as output:
        for degree in args.degrees:
            for n in args.sizes:
                space = Space(Mesh(n, jitter=0.18, seed=712), degree)
                initial = space.project(
                    lambda points, t: cp.ones((*points.shape[:-1], 1)), degree=20
                )
                references = []
                for subdivisions in (4, 8):
                    q = quadrature(20, subdivisions)
                    basis = space.basis(q[:, :2])
                    points = space.mesh.points(q)
                    rate = 1 + amplitude * cp.sin(2 * cp.pi * points[..., 0])
                    exact = cp.exp(-strength * rate)
                    references.append(
                        cp.einsum("cq,qb,q->cb", exact, basis, 2 * q[:, 2])[:, None, :]
                    )
                best = references[-1]
                norm = float(cp.sqrt(space.energy(best)[0]))
                ref_error = float(cp.sqrt(space.energy(references[0] - best)[0]) / norm)
                assert ref_error < 1e-10, ref_error
                # Exact time integration of the coarse Galerkin reaction ODE.
                rate_matrix = cp.einsum(
                    "cq,qi,qj,q->cij", rate, basis, basis, 2 * q[:, 2]
                )
                eigenvalues, eigenvectors = cp.linalg.eigh(rate_matrix)
                rotated = cp.einsum("cij,cfi->cfj", eigenvectors, initial)
                coarse = cp.einsum(
                    "cij,cfj->cfi",
                    eigenvectors,
                    rotated * cp.exp(-strength * eigenvalues[:, None, :]),
                )
                carrier = Carrier(space, n=args.naux, degree=4, builder="tiled")
                initial_energy = space.energy(initial)

                def record(method, steps, value, elapsed, setup, energy=None):
                    error = float(cp.sqrt(space.energy(value - best)[0]) / norm)
                    mass_error = float(
                        cp.abs(space.mass(value)[0] - space.mass(best)[0])
                        / cp.abs(space.mass(best)[0])
                    )
                    result = dict(
                        nfe=n,
                        naux=args.naux,
                        degree=degree,
                        aux_degree=4,
                        steps=steps,
                        strength=strength,
                        amplitude=amplitude,
                        method=method,
                        relative_observation_error=error,
                        relative_mass_error=mass_error,
                        reference_fe_norm=norm,
                        reference_mass=float(space.mass(best)[0]),
                        reference_quadrature_error=ref_error,
                        evolution_seconds=elapsed,
                        weighted_setup_seconds=setup,
                        positive_energy=energy,
                        quadrature_subdivisions=args.subdivisions,
                    )
                    output.write(
                        json.dumps(dict(result=result, provenance=meta)) + "\n"
                    )
                    output.flush()
                    print(json.dumps(result), flush=True)
                    records.append(result)

                record("fe-exponential", 0, coarse, None, None)
                for steps in args.steps:
                    begin = time.perf_counter()
                    reaction = ReactionMoments(
                        carrier,
                        strength / steps,
                        amplitude=amplitude,
                        builder="tiled",
                        subdivisions=args.subdivisions,
                    )
                    cp.cuda.Stream.null.synchronize()
                    setup = time.perf_counter() - begin
                    for method in args.methods:
                        state = MemoryTransport(carrier, initial)
                        auxiliary = carrier.project(initial)
                        cp.cuda.Stream.null.synchronize()
                        begin = time.perf_counter()
                        for _ in range(steps):
                            if method in ("full", "target"):
                                reaction.apply(
                                    state,
                                    iterations=1 if method == "target" else 2,
                                    target_seed=method == "target",
                                )
                            elif method == "reset":
                                state.apply_physics(reaction.apply_fe, hidden_decay=0.0)
                            else:
                                auxiliary = reaction.apply_auxiliary(auxiliary)
                                value = carrier.adjoint(auxiliary)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter() - begin
                        if method != "auxiliary":
                            value = state.value
                            energy = float(
                                ResidualEnergy(carrier).apply(
                                    state.anchor,
                                    carrier.project(state.anchor) - state.auxiliary,
                                )[0]
                            )
                        else:
                            energy = float(carrier.energy(auxiliary)[0])
                        assert energy <= float(
                            initial_energy[0]
                            * cp.exp(cp.asarray(-2 * strength * (1 - amplitude)))
                        ) * (1 + 1e-8)
                        record(method, steps, value, elapsed, setup, energy)
                    del reaction, state, auxiliary
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
