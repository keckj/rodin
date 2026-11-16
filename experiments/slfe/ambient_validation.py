"""Independent common-ambient-space oracle for polynomial memory projection.

Distributed under the Boost Software License, Version 1.0.
GPU Gram factorization embeds V, W0, W1 in one Euclidean space. A spectral
filter oracle checks the root-free production update and its contractions.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_grid import TensorGrid
from adaptive_study import partitions
from memory_transport import MemoryTransport
from campaign import provenance


def validate(p):
    space = Space(GradedMesh(4, 0.5), p)
    c0 = Carrier(space, edges=partitions(4, 0.8, cp.asarray([0.35, 0.45])))
    c1 = Carrier(space, edges=partitions(4, 0.7, cp.asarray([0.74, 0.81])))
    grid = TensorGrid(c1, c1.edges)
    remap = lambda w: grid.remap(c0, w)
    mf = cp.repeat(space.mesh.areas, space.nb)
    nv, nw = len(mf), c0.n**2 * c0.nc
    columns_v = lambda a: cp.ascontiguousarray(
        (a / cp.sqrt(mf)[:, None])
        .reshape(space.mesh.cells, space.nb, -1)
        .transpose(0, 2, 1)
    )
    vector_v = lambda a: cp.sqrt(mf)[:, None] * a.transpose(0, 2, 1).reshape(nv, -1)
    columns_w = lambda c, w: cp.ascontiguousarray(
        (w / cp.sqrt(c.mass.ravel())[:, None])
        .reshape(c.n**2, c.nc, -1)
        .transpose(0, 2, 1)
    )
    vector_w = lambda c, w: cp.sqrt(c.mass.ravel())[:, None] * w.transpose(
        0, 2, 1
    ).reshape(nw, -1)
    r0 = vector_w(c0, c0.project(columns_v(cp.eye(nv))))
    r1 = vector_w(c1, c1.project(columns_v(cp.eye(nv))))
    cross = vector_w(c1, remap(columns_w(c0, cp.eye(nw))))
    gram = cp.concatenate(
        (
            cp.concatenate((cp.eye(nv), r0.T, r1.T), axis=1),
            cp.concatenate((r0, cp.eye(nw), cross.T), axis=1),
            cp.concatenate((r1, cross, cp.eye(nw)), axis=1),
        ),
        axis=0,
    )
    eigen, vectors = cp.linalg.eigh(gram)
    minimum_gram = float(eigen.min())
    assert minimum_gram > -2e-10, minimum_gram
    root = (vectors * cp.sqrt(cp.maximum(eigen, 0))) @ vectors.T
    v, w0, w1 = root[:, :nv], root[:, nv : nv + nw], root[:, nv + nw :]
    h = w1 - v @ r1.T
    eigen_k, vectors_k = cp.linalg.eigh(cp.eye(nw) - r1 @ r1.T)
    assert float(eigen_k.min()) > -2e-10 and float(eigen_k.max()) < 1 + 2e-10
    eigen_k = cp.clip(eigen_k, 0, 1)
    random = cp.random.RandomState(921)
    original = MemoryTransport(c0, columns_v(random.normal(size=(nv, 3))))
    original.advect(c0.transport(0.17, "shear").apply_carrier)
    original.apply_physics(lambda z: 0.9 * z)
    original.advect(c0.transport(-0.07, "translation").apply_carrier)
    observed = vector_v(original.value)
    q = w0 @ vector_w(c0, original.auxiliary) + (v - w0 @ r0) @ vector_v(
        original.anchor
    )
    before = original.energy()
    previous = space.energy(original.value)
    rows = []
    for count in (1, 2, 4, 8, 16, 32):
        state = MemoryTransport(c0, original.value)
        state.anchor = original.anchor.copy()
        state.auxiliary = original.auxiliary.copy()
        loss = state.project_memory(c1, remap, count)
        qnew = w1 @ vector_w(c1, state.auxiliary) + (v - w1 @ r1) @ vector_v(
            state.anchor
        )
        polynomial = sum((1 - eigen_k) ** j for j in range(count))
        filtered = (vectors_k * polynomial) @ vectors_k.T
        oracle = v @ observed + h @ filtered @ h.T @ (q - v @ observed)
        state_error = float(cp.max(cp.abs(qnew - oracle)))
        observation_error = float(cp.max(cp.abs(v.T @ qnew - observed)))
        energy_error = float(
            cp.max(cp.abs(state.energy() - cp.sum(qnew * qnew, axis=0)))
        )
        minimum_gain = float(cp.min(state.energy() - previous))
        assert bool(cp.array_equal(state.value, original.value))
        assert max(state_error, observation_error, energy_error) < 2e-9
        assert float(cp.min(loss)) > -2e-9 and minimum_gain > -2e-9
        # Eigenvalues of the ambient hidden filter are analytically
        # 1-(1-lambda)^k, checked numerically without a random-vector proxy.
        filter_values = eigen_k * polynomial
        assert float(filter_values.max()) <= 1 + 1e-12
        rows.append(
            dict(
                iterations=count,
                state_error=state_error,
                observation_error=observation_error,
                energy_error=energy_error,
                retained_hidden_energy=(
                    state.energy() - space.energy(state.value)
                ).tolist(),
                loss=loss.tolist(),
                minimum_gain=minimum_gain,
            )
        )
        previous = state.energy()
    return dict(
        p=p,
        minimum_joint_gram=minimum_gram,
        initial_energy=before.tolist(),
        results=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for p in (1, 2, 4):
        records.append(validate(p))
        print(json.dumps(records[-1]), flush=True)
    args.output.write_text(
        json.dumps(dict(results=records, provenance=provenance()), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
