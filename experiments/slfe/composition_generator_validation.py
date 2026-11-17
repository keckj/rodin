"""GPU one-sided generator diagnostic for signed projected translations.

Distributed under the Boost Software License, Version 1.0.
For a projected translation S(t)=I+t*A-|t|*D+o(|t|), a composition
with sum(w)=1 has generator A-sum(abs(w))*D. Higher continuous order
does not remove the spatial jump dissipation at fixed FE resolution.
Dense GPU finite differences independently test this expansion.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_remap import TensorTransfer
from tensor_hamiltonian import ComposedHamiltonian
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for degree in (1, 2, 4):
        carrier = Carrier(Space(Mesh(4), degree), n=3, builder="tiled")
        nw = carrier.n**2 * carrier.nc
        identity = cp.ascontiguousarray(
            cp.eye(nw).reshape(carrier.n**2, carrier.nc, nw).transpose(0, 2, 1)
        )
        mass = cp.broadcast_to(carrier.mass, (carrier.n**2, carrier.nc)).ravel()
        scaling = cp.sqrt(mass[:, None] / mass[None, :])
        speed = cp.ones(carrier.n * carrier.np)

        def matrix(value):
            return value.transpose(0, 2, 1).reshape(nw, nw) * scaling

        def translate(value, dt):
            return TensorTransfer(carrier, [(0, dt, speed)]).apply_carrier(value)

        eye = cp.eye(nw)
        for scheme in ("triple4", "suzuki4", "yoshida6"):
            # Obtain the coefficients from the actual implemented composition.
            candidate = ComposedHamiltonian(
                carrier, [(1, 0)], cp.asarray([[0.013, 0.007]]), 0.1, scheme
            )
            weights = candidate.weights
            alpha = sum(abs(w) for w in weights)
            del candidate
            for dt in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                positive = matrix(translate(identity, dt))
                negative = matrix(translate(identity, -dt))
                skew = (positive - negative) / (2 * dt)
                dissipation = (2 * eye - positive - negative) / (2 * dt)
                result = identity.copy()
                for weight in weights:
                    result = translate(result, dt * weight)
                generator = (matrix(result) - eye) / dt
                expected = skew - alpha * dissipation
                residual = float(
                    cp.linalg.norm(generator - expected) / cp.linalg.norm(expected)
                )
                measured = float(-cp.trace(generator) / cp.trace(dissipation))
                r = dict(
                    degree=degree,
                    scheme=scheme,
                    duration=dt,
                    absolute_weight_sum=alpha,
                    measured_dissipation_multiplier=measured,
                    generator_relative_residual=residual,
                    symmetric_skew_defect=float(cp.linalg.norm(skew + skew.T)),
                    minimum_dissipation_eigenvalue=float(
                        cp.linalg.eigvalsh((dissipation + dissipation.T) / 2)[0]
                    ),
                )
                records.append(r)
                print(json.dumps(r), flush=True)
            assert residual < 1e-4 and abs(measured - alpha) < 1e-4, (
                degree,
                scheme,
                residual,
                measured,
                alpha,
            )
    args.output.write_text(
        json.dumps(dict(provenance=provenance(), controls=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
