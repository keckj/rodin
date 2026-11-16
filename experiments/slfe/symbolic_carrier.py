"""Exact Symox generation of 1D modal DG remap polynomials in Bernstein form.

Distributed under the Boost Software License, Version 1.0.
Only symbolic preprocessing runs on the host. Runtime matrices are built on GPU.
"""

import json
from math import comb
from pathlib import Path

import symox as sx


def generate(path):
    x, f = sx.symbols("x f")
    # Rational shifted Legendre polynomials. The modal normalization is applied
    # after integration, keeping all polynomial manipulations in exact rationals.
    basis = [sx.Integer(1), 2 * x - 1]
    for k in range(1, 5):
        basis.append(
            sx.expand(
                ((2 * k + 1) * (2 * x - 1) * basis[k] - k * basis[k - 1]) / (k + 1)
            )
        )
    tables = {}
    for degree in range(6):
        order = 2 * degree + 1
        matrix = []
        checks = 0
        for side in range(2):
            rows = []
            for i in range(degree + 1):
                row = []
                for j in range(degree + 1):
                    polynomial = sx.Poly(
                        sx.expand(basis[i] * basis[j].subs(x, x - f + (1 - side))), x
                    )
                    primitive = sum(
                        coefficient * x ** (power[0] + 1) / sx.Integer(power[0] + 1)
                        for power, coefficient in polynomial.terms()
                    )
                    integral = (
                        primitive.subs(x, f) - primitive.subs(x, 0)
                        if side == 0
                        else primitive.subs(x, 1) - primitive.subs(x, f)
                    )
                    power = sx.Poly(sx.expand(integral), f)
                    bernstein = [
                        sx.simplify(
                            sum(
                                power.nth(k) * sx.Rational(comb(d, k), comb(order, k))
                                for k in range(d + 1)
                            )
                        )
                        for d in range(order + 1)
                    ]
                    reconstructed = sum(
                        bernstein[k] * comb(order, k) * f**k * (1 - f) ** (order - k)
                        for k in range(order + 1)
                    )
                    assert sx.expand(reconstructed - integral) == 0
                    # Endpoint projections are the zero and identity matrices.
                    for endpoint in (0, 1):
                        expected = (
                            sx.Rational(int(i == j), 2 * i + 1)
                            if endpoint == 1 - side
                            else 0
                        )
                        assert sx.simplify(integral.subs(f, endpoint) - expected) == 0
                    normalization = sx.sqrt((2 * i + 1) * (2 * j + 1))
                    row.append([float(value * normalization) for value in bernstein])
                    checks += 3
                rows.append(row)
            matrix.append(rows)
        # Source/target constants are exact modal identities of the two blocks.
        for d in range(order + 1):
            for i in range(degree + 1):
                assert (
                    abs(sum(matrix[side][i][0][d] for side in range(2)) - int(i == 0))
                    < 2e-14
                )
                assert (
                    abs(sum(matrix[side][0][i][d] for side in range(2)) - int(i == 0))
                    < 2e-14
                )
        tables[str(degree)] = dict(
            degree=degree,
            bernstein_degree=order,
            modal_coefficients=matrix,
            exact_checks=checks,
        )
    path.write_text(json.dumps(dict(engine="symox", tables=tables), indent=2) + "\n")


if __name__ == "__main__":
    generate(Path(__file__).with_name("carrier_polynomials.json"))
