"""Limit measured memory-transfer distance by backtracking a proposed mesh move.

Distributed under the Boost Software License, Version 1.0.
The Pythagorean interpretation requires exact ambient projections. The floating
point guard below is empirical; it is not an interval-arithmetic certificate.
The budget controls this transfer only, not the error of subsequent advection.
"""

from copy import copy

from gpu import cp
from carrier import Carrier
from tensor_grid import TensorGrid


def budgeted_move(state, proposed_edges, tolerance, maximum_backtracks=8):
    if tolerance <= 0 or maximum_backtracks < 0:
        raise ValueError(
            "Expected a positive tolerance and nonnegative backtrack count"
        )
    old = state.coupling
    guard = 1e-12 * max(1.0, float(cp.max(state.energy())))
    if tolerance**2 <= guard:
        return state, 0.0, None, []
    attempts = []
    for index in range(maximum_backtracks + 1):
        fraction = 2.0 ** (-index)
        edges = old.edges + fraction * (proposed_edges - old.edges)
        target = Carrier(old.space, n=old.n, degree=old.np - 1, edges=edges)
        grid = TensorGrid(target, target.edges)
        candidate = copy(state)
        diagnostic = candidate.project_memory_galerkin(
            target, lambda w: grid.remap(old, w), iterations=2
        )
        # Use both independently equivalent diagnostic expressions; neither
        # a negative roundoff loss nor a small inconsistent value grants credit.
        squared = float(
            cp.max(
                cp.maximum(
                    diagnostic["energy_loss"], diagnostic["ambient_change_squared"]
                )
            )
        )
        bound = (max(squared, 0.0) + guard) ** 0.5
        discrepancy = float(
            cp.max(
                cp.abs(diagnostic["energy_loss"] - diagnostic["ambient_change_squared"])
            )
        )
        attempts.append(
            dict(
                fraction=fraction,
                measured_distance=bound,
                pythagorean_discrepancy=discrepancy,
            )
        )
        if bound <= tolerance:
            return candidate, fraction, diagnostic, attempts
    return state, 0.0, None, attempts
