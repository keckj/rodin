"""Independent CUDA checks of lossless memory refinement on nested DG grids.

Distributed under the Boost Software License, Version 1.0.
Full Gram matrices check all FE initial conditions simultaneously. Projection
coarsening is checked separately; it is not passed to the lossless update.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_projection import TensorProjection
from tensor_study import partitions
from memory_transport import MemoryTransport
from campaign import provenance


def validate(p, local):
    space = Space(GradedMesh(4, 0.5), p)
    edges = partitions(4, 0.8)
    middle = (edges[:, :-1] + edges[:, 1:]) / 2
    if local:
        middle = middle[:, 1:3]
    finer = cp.sort(cp.concatenate((edges, middle), axis=1), axis=1)
    old = Carrier(space, n=4, edges=edges)
    new = Carrier(space, n=finer.shape[1] - 1, edges=finer)
    projection = TensorProjection(old, new)
    reverse = TensorProjection(new, old)
    nv = space.mesh.cells * space.nb
    nw = old.n**2 * old.nc
    mw0 = old.mass.ravel()
    mf = cp.repeat(space.mesh.areas, space.nb)
    v = cp.ascontiguousarray(
        (cp.eye(nv) / cp.sqrt(mf)[:, None])
        .reshape(space.mesh.cells, space.nb, nv)
        .transpose(0, 2, 1)
    )
    eye = cp.ascontiguousarray(
        (cp.eye(nw) / cp.sqrt(mw0)[:, None])
        .reshape(old.n**2, old.nc, nw)
        .transpose(0, 2, 1)
    )
    vector_w = lambda c, w: cp.sqrt(c.mass.ravel())[:, None] * w.transpose(
        0, 2, 1
    ).reshape(c.n**2 * c.nc, -1)
    vector_v = lambda v: cp.sqrt(mf)[:, None] * v.transpose(0, 2, 1).reshape(nv, -1)
    injection = vector_w(new, projection.apply_carrier(eye))
    isometry = float(cp.max(cp.abs(injection.T @ injection - cp.eye(nw))))
    r0, r1 = vector_w(old, old.project(v)), vector_w(new, new.project(v))
    compatibility = float(cp.max(cp.abs(injection.T @ r1 - r0)))
    assert max(isometry, compatibility) < 4e-11
    ones = cp.ones((old.n**2, 1, old.nc))
    constant = float(cp.max(cp.abs(projection.apply_carrier(ones) - 1)))
    roundtrip = float(
        cp.max(cp.abs(reverse.apply_carrier(projection.apply_carrier(eye)) - eye))
    )
    assert constant < 3e-11 and roundtrip < 2e-10
    state = MemoryTransport(old, v)
    state.advect(old.transport(0.17, "shear").apply_carrier)

    def physics(v):
        constant = cp.zeros_like(v)
        constant[:, :, 0] = space.mass(v)[None, :]
        return 0.9 * v + 0.1 * constant

    state.apply_physics(physics)
    state.advect(old.transport(-0.07, "translation").apply_carrier)
    a0, w0 = vector_v(state.anchor), vector_w(old, state.auxiliary)
    gram0 = w0.T @ w0 + a0.T @ (cp.eye(nv) - r0.T @ r0) @ a0
    expected_v = state.value.copy()
    loss = state.refine(new, projection.apply_carrier)
    a1, w1 = vector_v(state.anchor), vector_w(new, state.auxiliary)
    gram1 = w1.T @ w1 + a1.T @ (cp.eye(nv) - r1.T @ r1) @ a1
    gram_error = float(cp.max(cp.abs(gram1 - gram0)))
    observed = r1.T @ w1 + (cp.eye(nv) - r1.T @ r1) @ a1
    observed_error = float(cp.max(cp.abs(observed - vector_v(expected_v))))
    assert bool(cp.array_equal(state.value, expected_v))
    assert max(gram_error, observed_error, float(cp.max(cp.abs(loss)))) < 8e-10
    # Independent sparse tensor projection also agrees with the established
    # row-remap kernels on a same-size, nonnested partition with common endpoints.
    alternative = edges.copy()
    alternative[:, 1:-1] = 0.9 * edges[:, 1:-1] + 0.1 * (
        edges[:, :1] + cp.arange(1, 4)[None, :] / 4
    )
    grid = TensorGrid(old, alternative)
    sparse = TensorProjection(old, grid).apply_carrier(eye)
    rowwise = grid.remap(old, eye)
    implementation_error = float(cp.max(cp.abs(sparse - rowwise)))
    assert implementation_error < 3e-10
    return dict(
        p=p,
        local=local,
        old_cells=old.n**2,
        new_cells=new.n**2,
        injection_isometry_error=isometry,
        compatibility_error=compatibility,
        constant_error=constant,
        roundtrip_error=roundtrip,
        gram_error=gram_error,
        observation_error=observed_error,
        energy_loss_max=float(cp.max(cp.abs(loss))),
        sparse_vs_rowwise_error=implementation_error,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for p in (1, 2, 4):
        for local in (False, True):
            record = validate(p, local)
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(dict(results=records, provenance=provenance()), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
