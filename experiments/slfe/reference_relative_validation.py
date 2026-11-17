"""GPU dense isometric oracles for contraction relative to FE reference states.

Distributed under the Boost Software License, Version 1.0.
Direct distances in the dilated Euclidean coordinates distinguish this weaker
property from pairwise nonexpansivity of a nonlinear memory transfer.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from memory_transport import MemoryTransport
from tensor_grid import TensorGrid
from reaction_moments_validation import joint_matrix
from campaign import provenance


def embedding(c):
    nv = c.space.mesh.cells * c.space.nb
    r = joint_matrix(c)[nv:, :nv]
    defect = cp.eye(nv) - r.T @ r
    eigen, vectors = cp.linalg.eigh((defect + defect.T) / 2)
    assert float(eigen.min()) > -1e-10
    root = (vectors * cp.sqrt(cp.maximum(eigen, 0))[None, :]) @ vectors.T
    return cp.concatenate((r, root)), root


def fe_coordinates(c, v):
    return (
        (v * cp.sqrt(c.space.mesh.areas)[:, None, None])
        .transpose(0, 2, 1)
        .reshape(-1, v.shape[1])
    )


def coordinates(state, root):
    c = state.coupling
    mass = cp.broadcast_to(c.mass, (c.n**2, c.nc))
    w = (
        (state.auxiliary * cp.sqrt(mass)[:, None, :])
        .transpose(0, 2, 1)
        .reshape(-1, state.value.shape[1])
    )
    return cp.concatenate((w, root @ fe_coordinates(c, state.anchor)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    results = []
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.18, seed=817), degree)
        old = Carrier(space, n=4, builder="tiled")
        random = cp.random.RandomState(415)
        initial = random.normal(size=(space.mesh.cells, 4, space.nb))
        original = MemoryTransport(old, initial)
        original.advect(old.transport(0.07, "shear").apply_carrier)
        original.apply_physics(lambda x: 0.9 * x)
        j0, d0 = embedding(old)
        before = coordinates(original, d0)
        references = [
            cp.zeros_like(initial),
            random.normal(size=initial.shape),
            100 * random.normal(size=initial.shape),
        ]
        for motion in (0.0, 0.02, 0.1):
            x = cp.linspace(0, 1, 5)
            edges = cp.stack(
                (x + motion * cp.sin(2 * cp.pi * x) / (2 * cp.pi), x + motion / 3)
            )
            new = Carrier(space, n=4, edges=edges, builder="tiled")
            grid = TensorGrid(new, new.edges)
            cross = lambda value: grid.remap(old, value)
            j1, d1 = embedding(new)
            for method in ("reset", "chebyshev", "warm", "galerkin", "block"):
                state = copy(original)
                if method == "reset":
                    state.restart(new)
                elif method == "chebyshev":
                    state.project_memory(new, cross, iterations=4, filter_scale=0.1)
                elif method == "warm":
                    state.project_memory_warm(new, cross, iterations=8)
                else:
                    state.project_memory_galerkin(
                        new,
                        cross,
                        shared=method == "block",
                        gram_backend="reduction",
                    )
                after = coordinates(state, d1)
                loss = cp.sum(before**2 - after**2, axis=0)
                for index, reference in enumerate(references):
                    u = fe_coordinates(old, reference)
                    distance_before = cp.sum((before - j0 @ u) ** 2, axis=0)
                    distance_after = cp.sum((after - j1 @ u) ** 2, axis=0)
                    scale = cp.maximum(distance_before, 1)
                    defect = float(
                        cp.max(cp.abs(distance_before - distance_after - loss) / scale)
                    )
                    excess = float(cp.max((distance_after - distance_before) / scale))
                    assert defect < 2e-11 and excess < 2e-11, (defect, excess)
                    results.append(
                        dict(
                            degree=degree,
                            motion=motion,
                            method=method,
                            reference=index,
                            relative_identity_defect=defect,
                            relative_distance_excess=excess,
                            energy_loss=loss.tolist(),
                        )
                    )
        args.output.write_text(
            json.dumps(dict(results=results, provenance=meta), indent=2) + "\n"
        )
        print(json.dumps(dict(degree=degree, controls=len(results))), flush=True)


if __name__ == "__main__":
    main()
