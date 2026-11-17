"""Independent finite-dimensional stability counterexamples, entirely on CUDA.

Distributed under the Boost Software License, Version 1.0.
Energy contraction and the projection identity do not imply contraction of
differences for a state-dependent augmented Galerkin transfer.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from campaign import provenance


def check(directions):
    eigenvalues = cp.asarray([0.01, 0.1, 1.0])
    # V=span(e0,e1,e2), Wold=V-perp. The new auxiliary basis is
    # cos(theta_i)e_i + sin(theta_i)e_(i+3). Thus K has the stated spectrum.
    visible = cp.eye(6)[:, :3]
    old = cp.eye(6)[:, 3:]
    new = visible * cp.sqrt(1 - eigenvalues) + old * cp.sqrt(eigenvalues)
    projection = visible @ visible.T
    hidden = (cp.eye(6) - projection) @ new
    gram_operator = hidden.T @ hidden
    assert float(cp.max(cp.abs(gram_operator - cp.diag(eigenvalues)))) < 1e-14
    initial = cp.asarray([1.0, 0.0, 0.0] if directions == 1 else [1.0, 1.0, 0.0])
    initial /= cp.linalg.norm(initial)
    epsilon = 1e-7
    perturbed = initial.copy()
    perturbed[-1] += epsilon
    mapped, diagnostics = [], []
    for coefficients in (initial, perturbed):
        q = old @ coefficients
        h = new.T @ q
        # Here the incoming direction equals h because Rold=0.
        # Match the production sequence [mapped, h, R R* h, ...].
        vectors = [h]
        candidate = h
        for _ in range(directions - 1):
            vectors.append(candidate)
            candidate = (cp.eye(3) - gram_operator) @ candidate
        basis = cp.stack(vectors, axis=1)
        basis /= cp.linalg.norm(basis, axis=0)[None, :]
        gram = basis.T @ gram_operator @ basis
        eigen, rotation = cp.linalg.eigh((gram + gram.T) / 2)
        cutoff = 1e-12 * cp.maximum(1.0, eigen[-1])
        inverse = cp.where(eigen > cutoff, 1 / cp.maximum(eigen, cutoff), 0)
        x = basis @ (rotation @ (inverse * (rotation.T @ (basis.T @ h))))
        result = hidden @ x
        loss = cp.dot(q, q) - cp.dot(result, result)
        distance = cp.sum((q - result) ** 2)
        assert float(loss) > -1e-11
        assert float(cp.abs(loss - distance)) < 1e-10
        mapped.append(result)
        diagnostics.append(
            dict(energy_loss=float(loss), distance_squared=float(distance))
        )
    gain = float(cp.linalg.norm(mapped[1] - mapped[0]) / epsilon)
    # For one effective direction, g(lambda)=lambda/lambda_1.
    # For two, g(lambda)=1-(1-lambda/lambda_1)(1-lambda/lambda_2).
    predicted = 100.0 if directions == 1 else 890.0
    assert abs(gain / predicted - 1) < 1e-4, (gain, predicted)
    return dict(
        directions=directions,
        effective_directions=1 if directions == 1 else 2,
        eigenvalues=eigenvalues.tolist(),
        initial=initial.tolist(),
        perturbed=perturbed.tolist(),
        epsilon=epsilon,
        distance_gain=gain,
        limiting_gain=predicted,
        diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(results=[check(1), check(3)], provenance=provenance())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
