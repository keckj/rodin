"""GPU diagnostic of raw-target retention after the existing Gram rank cutoff.

Distributed under the Boost Software License, Version 1.0.
The diagnostic repeats a projection with exactly the realized direction
space, but uses its raw target moment as the right-hand side. Positive
physical quadrature measures whether numerical truncation still retains it.
These instrumented runs are excluded from all performance comparisons.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space, quadrature
from carrier import Carrier
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from residual_energy import ResidualEnergy
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    space = Space(Mesh(8, jitter=0.18, seed=712), 4)
    carrier = Carrier(space, n=128, degree=4, builder="tiled")
    initial = space.project(
        lambda points, t: cp.ones((*points.shape[:-1], 1)), degree=20
    )
    state = MemoryTransport(carrier, initial)
    reaction = ReactionMoments(
        carrier, 20 / 64, amplitude=0.95, builder="tiled", subdivisions=2
    )
    energy = ResidualEnergy(carrier)
    q = quadrature(20, 8)
    basis = space.basis(q[:, :2])
    rate = 1 + 0.95 * cp.sin(2 * cp.pi * space.mesh.points(q)[..., 0])
    records = []
    ranks = []
    for step in range(1, 65):
        incoming = state.auxiliary - carrier.project(state.anchor)
        moment = reaction.cross[1].project(state.anchor) + reaction.apply_auxiliary(
            incoming
        )
        result = reaction.apply(state, iterations=1, target_seed=True)
        ranks.append(int(result["retained_rank"]))
        if step not in (1, 2, 3, 4, 16, 32, 64):
            continue
        observable = carrier.adjoint(moment)
        probe = MemoryTransport(carrier, cp.zeros_like(initial))
        probe.project_moments(
            carrier,
            observable,
            moment - carrier.project(observable),
            incoming,
            iterations=1,
            shared=True,
            search_seed=moment,
            gram_backend="reduction",
        )
        detail = probe.auxiliary - carrier.project(probe.anchor)
        lost = energy.apply(probe.anchor, moment - detail)
        relative = float(cp.sqrt(lost[0] / carrier.energy(moment)[0]))
        reference = cp.einsum(
            "cq,qb,q->cb", cp.exp(-20 * step / 64 * rate), basis, 2 * q[:, 2]
        )[:, None, :]
        error = float(
            cp.sqrt(
                space.energy(state.value - reference)[0] / space.energy(reference)[0]
            )
        )
        record = dict(
            step=step,
            retained_rank=ranks[-1],
            gram_eigenvalues=result["gram_eigenvalues"].tolist(),
            physical_relative_target_retention_error=relative,
            relative_observation_error=error,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                provenance=provenance(),
                degree=4,
                nfe=8,
                naux=128,
                steps=64,
                cutoff="1e-12 * max(largest Gram eigenvalue, 1)",
                ranks=ranks,
                controls=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
