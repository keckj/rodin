"""Independent lifted validation of a root-free hidden-memory transfer.

Distributed under the Boost Software License, Version 1.0.
The transfer B=J1 J0*+Q1 diag(C,0) Q0 preserves the FE observation and
contracts energy. Dense roots and singular values occur only in this oracle.
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
    target = TensorGrid(c1, c1.edges)
    mf = cp.repeat(space.mesh.areas, space.nb)
    nv, nw = len(mf), c0.n**2 * c0.nc
    mv = cp.sqrt(mf)
    columns_v = lambda v: cp.ascontiguousarray(
        (v / mv[:, None]).reshape(space.mesh.cells, space.nb, -1).transpose(0, 2, 1)
    )
    vector_v = lambda v: mv[:, None] * v.transpose(0, 2, 1).reshape(nv, -1)
    vector_w = lambda c, w: cp.sqrt(c.mass.ravel())[:, None] * w.transpose(
        0, 2, 1
    ).reshape(nw, -1)
    columns_w = lambda c, w: cp.ascontiguousarray(
        (w / cp.sqrt(c.mass.ravel())[:, None])
        .reshape(c.n**2, c.nc, -1)
        .transpose(0, 2, 1)
    )
    roots, lifts = [], []
    for c in (c0, c1):
        r = vector_w(c, c.project(columns_v(cp.eye(nv))))
        e, vectors = cp.linalg.eigh(cp.eye(nv) - r.T @ r)
        assert float(e.min()) > -3e-11
        d = (vectors * cp.sqrt(cp.maximum(e, 0))) @ vectors.T
        roots.append(d)
        lifts.append(cp.concatenate((r, d), axis=0))
    j0, j1 = lifts
    q0, q1 = cp.eye(nv + nw) - j0 @ j0.T, cp.eye(nv + nw) - j1 @ j1.T
    c = vector_w(c1, target.remap(c0, columns_w(c0, cp.eye(nw))))
    b = j1 @ j0.T + q1[:, :nw] @ c @ q0[:nw, :]
    sv = float(cp.linalg.svd(b, compute_uv=False)[0])
    fe_consistency = float(cp.max(cp.abs(j1.T @ b - j0.T)))
    assert sv < 1 + 5e-11 and fe_consistency < 5e-11
    initial = columns_v(cp.random.RandomState(711).normal(size=(nv, 3)))
    state = MemoryTransport(c0, initial)
    flow = c0.transport(0.17, "shear")
    state.advect(flow.apply_carrier)

    def physics(v):
        constant = cp.zeros_like(v)
        constant[:, :, 0] = space.mass(v)[None, :]
        return 0.9 * v + 0.1 * constant

    state.apply_physics(physics)
    state.advect(flow.apply_carrier)
    z0 = cp.concatenate(
        (vector_w(c0, state.auxiliary), roots[0] @ vector_v(state.anchor)), axis=0
    )
    v0 = state.value.copy()
    r = vector_w(c0, state.auxiliary - c0.project(v0))
    h = c @ r
    term1 = cp.sum((roots[0] @ vector_v(state.anchor - v0)) ** 2, axis=0)
    term2 = cp.sum(r * r, axis=0) - cp.sum(h * h, axis=0)
    term3 = cp.sum((j1[:nw].T @ h) ** 2, axis=0)
    reset_loss = state.energy() - space.energy(v0)
    measured_loss = state.transfer_memory(c1, lambda w: target.remap(c0, w))
    z1 = cp.concatenate(
        (vector_w(c1, state.auxiliary), roots[1] @ vector_v(state.anchor)), axis=0
    )
    state_error = float(cp.max(cp.abs(z1 - b @ z0)))
    loss_error = float(cp.max(cp.abs(measured_loss - term1 - term2 - term3)))
    observation_error = float(cp.max(cp.abs(j1.T @ z1 - vector_v(v0))))
    assert max(state_error, loss_error, observation_error) < 3e-9
    assert bool(cp.array_equal(state.value, v0))
    assert float(cp.min(measured_loss)) > -1e-10
    assert float(cp.max(measured_loss - reset_loss)) < 1e-10
    return dict(
        p=p,
        singular_max=sv,
        fe_consistency=fe_consistency,
        state_error=state_error,
        loss_identity_error=loss_error,
        observation_error=observation_error,
        transfer_loss=measured_loss.tolist(),
        reset_loss=reset_loss.tolist(),
        loss_terms=[x.tolist() for x in (term1, term2, term3)],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [validate(p) for p in (1, 2, 4)]
    args.output.write_text(
        json.dumps(dict(results=records, provenance=provenance()), indent=2) + "\n"
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
