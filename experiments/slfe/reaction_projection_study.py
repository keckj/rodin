"""Compare compressed physical moments with the full ambient projection on GPU.

Distributed under the Boost Software License, Version 1.0.
The oracle solves the dense joint Gram problem, independent of the small
hidden-space implementation. This is a projection study, not a PDE benchmark.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from reaction_moments_validation import joint_matrix, coordinates
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", choices=("moment", "increment", "residual"), default="moment"
    )
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for degree in (1, 2, 4):
            space = Space(Mesh(4, jitter=0.18, seed=817), degree)
            coupling = Carrier(space, n=4)
            state = MemoryTransport(coupling, cp.zeros((space.mesh.cells, 4, space.nb)))
            rng = cp.random.RandomState(141)
            state.anchor = rng.normal(size=state.anchor.shape)
            d = rng.normal(size=state.auxiliary.shape)
            state.auxiliary = coupling.project(state.anchor) + d
            state.value = state.anchor + coupling.adjoint(d)
            c = coordinates(state)
            gram = joint_matrix(coupling)
            eigen, vectors = cp.linalg.eigh((gram + gram.T) / 2)
            cutoff = 1e-12 * float(eigen.max())
            inverse = cp.where(eigen > cutoff, 1 / cp.maximum(eigen, cutoff), 0)
            for duration in (0.01, 0.2, 1.0):
                operator = ReactionMoments(coupling, duration, diagnostics=True)
                weighted = joint_matrix(coupling, operator)
                squared = joint_matrix(coupling, operator, 2)
                rhs = weighted @ c
                target_energy = cp.sum(c * (squared @ c), axis=0)
                best = vectors @ (inverse[:, None] * (vectors.T @ rhs))
                best_energy = cp.sum(best * (gram @ best), axis=0)
                best_loss = target_energy - best_energy
                visible_energy = space.energy(
                    operator.apply_fe(state.anchor) + operator.cross[1].adjoint(d)
                )
                for shared in (False, True):
                    previous = None
                    for iterations in (0, 1, 2, 4, 8):
                        result = copy(state)
                        diagnostic = operator.apply(
                            result,
                            iterations=iterations,
                            shared=shared,
                            krylov_seed=args.seed,
                        )
                        new = coordinates(result)
                        energy = cp.sum(new * (gram @ new), axis=0)
                        distance = (
                            target_energy + energy - 2 * cp.sum(new * rhs, axis=0)
                        )
                        pythagoras_error = float(
                            cp.max(cp.abs(distance - (target_energy - energy)))
                        )
                        assert pythagoras_error < 1e-9, pythagoras_error
                        excess = float(cp.max(energy - best_energy))
                        monotonic_defect = (
                            0.0
                            if previous is None
                            else float(cp.max(previous - energy))
                        )
                        assert excess < 1e-8 and monotonic_defect < 1e-8, (
                            excess,
                            monotonic_defect,
                        )
                        previous = energy
                        record = dict(
                            degree=degree,
                            duration=duration,
                            fields=4,
                            krylov_seed=args.seed,
                            pythagoras_error=pythagoras_error,
                            shared=shared,
                            directions_per_field=iterations + 1,
                            full_gram_rank=int(cp.sum(eigen > cutoff)),
                            small_gram_rank=diagnostic["retained_rank"].tolist(),
                            full_projection_loss=best_loss.tolist(),
                            compressed_projection_loss=distance.tolist(),
                            relative_hidden_energy_retained=(
                                (energy - visible_energy)
                                / (best_energy - visible_energy)
                            ).tolist(),
                            excess_above_full=excess,
                            enrichment_monotonic_defect=monotonic_defect,
                        )
                        stream.write(
                            json.dumps(dict(result=record, provenance=meta)) + "\n"
                        )
                        stream.flush()
                print(json.dumps(dict(degree=degree, duration=duration)), flush=True)
            del operator, coupling, state
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
