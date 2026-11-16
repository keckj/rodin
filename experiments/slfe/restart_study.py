"""Validate variable auxiliary spaces against explicit lifted GPU operators.

Distributed under the Boost Software License, Version 1.0.
Dense square roots occur only in this independent validation oracle.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from fourier_carrier import FourierCarrier, UnitaryCellular
from tensor_study import partitions
from memory_transport import MemoryTransport
from campaign import provenance


def validate(p=2):
    space = Space(GradedMesh(4, 0.5), p)
    mf = cp.repeat(space.mesh.areas, space.nb)
    nv = len(mf)

    def columns(v):
        return cp.ascontiguousarray(
            (v / cp.sqrt(mf)[:, None])
            .reshape(space.mesh.cells, space.nb, -1)
            .transpose(0, 2, 1)
        )

    def vector(v):
        return cp.sqrt(mf)[:, None] * v.transpose(0, 2, 1).reshape(nv, -1)

    carriers = [
        Carrier(space, n=4, edges=partitions(4, 0.8)),
        FourierCarrier(space, modes=7),
        Carrier(space, n=5, edges=partitions(5, 0.6)),
    ]
    lifted = []
    operations = []
    matrices = []
    weights = []
    for c in carriers:
        if isinstance(c, FourierCarrier):
            mw = cp.full(c.n**2, 1 / c.n**2)
            raw = lambda w: w.reshape(c.n * c.n, -1)

            def columns_w(w, c=c):
                return (w * c.n).reshape(c.n, c.n, -1)

            flow = UnitaryCellular(c.n)
            operation = lambda w, flow=flow: flow.step(w, 0.037)
        else:
            mw = c.mass.ravel()

            def raw(w):
                return w.transpose(0, 2, 1).reshape(-1, w.shape[1])

            def columns_w(w, c=c, mw=mw):
                return cp.ascontiguousarray(
                    (w / cp.sqrt(mw)[:, None])
                    .reshape(c.n * c.n, c.nc, -1)
                    .transpose(0, 2, 1)
                )

            transport = c.transport(0.037, "shear")
            operation = transport.apply_carrier
        r = cp.sqrt(mw)[:, None] * raw(c.project(columns(cp.eye(nv))))
        eigen, vectors = cp.linalg.eigh(cp.eye(nv) - r.T @ r)
        assert float(eigen.min()) > -3e-11
        d = (vectors * cp.sqrt(cp.maximum(eigen, 0))) @ vectors.T
        j = cp.concatenate((r, d), axis=0)
        assert float(cp.max(cp.abs(j.T @ j - cp.eye(nv)))) < 5e-11
        nw = len(mw)
        s = cp.sqrt(mw)[:, None] * raw(operation(columns_w(cp.eye(nw))))
        assert float(cp.linalg.svd(s, compute_uv=False)[0]) < 1 + 3e-11
        lifted.append(j)
        operations.append(operation)
        matrices.append(s)
        weights.append(mw)

    initial = cp.random.RandomState(792).normal(size=(nv, 3))
    state = MemoryTransport(carriers[0], columns(initial))
    z = lifted[0] @ initial
    current = 0
    records = []
    for iteration in range(12):
        j, s = lifted[current], matrices[current]
        nw = s.shape[0]
        state.advect(operations[current])
        z[:nw] = s @ z[:nw]
        before = state.value

        def physics(v):
            constant = cp.zeros_like(v)
            constant[:, :, 0] = space.mass(v)[None, :]
            return 0.93 * v + 0.07 * constant

        state.apply_physics(physics)
        z += j @ vector(state.value - before)
        observation_error = float(cp.max(cp.abs(vector(state.value) - j.T @ z)))
        energy_error = float(cp.max(cp.abs(state.energy() - cp.sum(z * z, axis=0))))
        assert max(observation_error, energy_error) < 5e-9
        oldvalue = state.value.copy()
        observed = j.T @ z
        lost = cp.sum((z - j @ observed) ** 2, axis=0)
        following = (current + 1) % len(carriers)
        measured_loss = state.restart(carriers[following])
        z = lifted[following] @ observed
        loss_error = float(cp.max(cp.abs(measured_loss - lost)))
        assert loss_error < 5e-9
        assert bool(cp.array_equal(state.value, oldvalue))
        assert float(cp.max(cp.abs(state.energy() - space.energy(state.value)))) < 5e-9
        mass_error = float(
            cp.max(cp.abs(space.mass(state.value) - space.mass(columns(initial))))
        )
        assert mass_error < 5e-10
        assert float(cp.max(state.energy() - cp.sum(initial * initial, axis=0))) < 5e-9
        records.append(
            dict(
                iteration=iteration,
                source=current,
                target=following,
                observation_error=observation_error,
                energy_error=energy_error,
                loss_error=loss_error,
                loss=measured_loss.tolist(),
                mass_error=mass_error,
            )
        )
        current = following
    return dict(p=p, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for p in (1, 2, 4):
        results.append(validate(p))
        print(json.dumps(dict(p=p, passed=True)), flush=True)
    args.output.write_text(
        json.dumps(dict(results=results, provenance=provenance()), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
