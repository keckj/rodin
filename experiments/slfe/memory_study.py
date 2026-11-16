"""Independent lifted-matrix validation of FE/auxiliary coupling with memory.

Distributed under the Boost Software License, Version 1.0.
Dense GPU square roots and exponentials occur only in this validation oracle.
"""

import json
from pathlib import Path
import argparse

from cupyx.scipy.linalg import expm
from gpu import cp, Mesh, Space
from fourier_carrier import FourierCarrier, UnitaryCellular
from memory_transport import MemoryTransport
from campaign import provenance


def validate(p=2, factor=1, exact_auxiliary=True):
    space = Space(Mesh(4), p)
    coupling = FourierCarrier(space, factor=factor)
    flow = UnitaryCellular(coupling.n)
    nv = space.mesh.cells * space.nb
    nw = coupling.n**2
    mass = cp.repeat(space.mesh.areas, space.nb)
    sqrt_mass = cp.sqrt(mass)

    def columns(x):
        return cp.ascontiguousarray(
            x.reshape(space.mesh.cells, space.nb, -1).transpose(0, 2, 1)
        )

    def matrix(x):
        return x.transpose(0, 2, 1).reshape(nv, -1)

    r = (
        coupling.project(columns(cp.eye(nv) / sqrt_mass[:, None])).reshape(nw, nv)
        / coupling.n
    )
    defect = cp.eye(nv) - r.T @ r
    eigenvalues, eigenvectors = cp.linalg.eigh(defect)
    assert float(eigenvalues[0]) > -3e-11
    d = (eigenvectors * cp.sqrt(cp.maximum(eigenvalues, 0))[None, :]) @ eigenvectors.T
    j = cp.concatenate((r, d), axis=0)
    isometry = float(cp.max(cp.abs(j.T @ j - cp.eye(nv))))
    assert isometry < 3e-11
    l = flow.generator(cp.eye(nw).reshape(coupling.n, coupling.n, nw)).reshape(nw, nw)
    skew = float(cp.max(cp.abs(l + l.T)))
    assert skew < 3e-11
    one = cp.zeros(nv)
    one[:: space.nb] = cp.sqrt(space.mesh.areas)
    unit = one / cp.linalg.norm(one)
    initial = cp.random.RandomState(819).normal(size=(nv, 3))
    z0 = j @ initial
    results = []
    for damping in (1.0, 10.0):
        lf = -damping * (cp.eye(nv) - unit[:, None] * unit[None, :])
        generator = j @ lf @ j.T
        generator[:nw, :nw] += l
        reference = expm(0.2 * generator) @ z0
        errors = []
        for steps in (1, 2, 4, 8, 16, 32) if exact_auxiliary else (16, 32, 64, 128):
            state = MemoryTransport(coupling, columns(initial / sqrt_mass[:, None]))
            z = d @ initial
            dt = 0.2 / steps
            exact_transport = expm(dt * l)
            decay = cp.exp(cp.asarray(-damping * dt / 2))
            energy0 = cp.sum(z0 * z0, axis=0)
            maximum_growth = 0.0
            for _ in range(steps):
                for stage in ("physics", "advection", "physics"):
                    before = state.value
                    if stage == "physics":

                        def relaxation(value):
                            means = space.mass(value)
                            constant = cp.zeros_like(value)
                            constant[:, :, 0] = means[None, :]
                            return constant + decay * (value - constant)

                        state.apply_physics(relaxation)
                        z += d @ (sqrt_mass[:, None] * matrix(state.value - before))
                    else:
                        if exact_auxiliary:
                            state.advect(
                                lambda value: (
                                    exact_transport @ value.reshape(nw, -1)
                                ).reshape(value.shape)
                            )
                        else:
                            state.advect(lambda value: flow.step(value, dt))
                    lifted = cp.concatenate(
                        (state.auxiliary.reshape(nw, -1) / coupling.n, z), axis=0
                    )
                    energy = cp.sum(lifted * lifted, axis=0)
                    maximum_growth = max(
                        maximum_growth, float(cp.max(energy - energy0))
                    )
                    energy0 = energy
            observed = sqrt_mass[:, None] * matrix(state.value)
            observation_error = float(cp.max(cp.abs(observed - j.T @ lifted)))
            mass_error = float(cp.max(cp.abs(unit @ observed - unit @ initial)))
            error = float(
                cp.linalg.norm(lifted - reference) / cp.linalg.norm(reference)
            )
            assert (
                observation_error < 2e-10
                and mass_error < 2e-10
                and maximum_growth < 2e-9
            )
            errors.append(
                dict(
                    steps=steps,
                    error=error,
                    observation_error=observation_error,
                    mass_error=mass_error,
                    maximum_energy_growth=maximum_growth,
                )
            )
        ratios = [
            errors[k]["error"] / errors[k + 1]["error"] for k in range(len(errors) - 1)
        ]
        assert 3.7 < ratios[-1] < 4.3, ratios
        results.append(dict(damping=damping, errors=errors, refinement_ratios=ratios))
    return dict(
        p=p,
        factor=factor,
        exact_auxiliary=exact_auxiliary,
        isometry_error=isometry,
        skew_error=skew,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for p, factor in ((1, 1), (2, 1), (2, 2), (3, 1)):
        record = dict(status="ok", result=validate(p, factor), provenance=provenance())
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(dict(p=p, factor=factor, status="ok")), flush=True)


if __name__ == "__main__":
    main()
