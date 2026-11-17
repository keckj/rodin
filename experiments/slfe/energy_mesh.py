"""GPU equidistribution monitors for several localized transported fields.

Distributed under the Boost Software License, Version 1.0.
The smoothed energy monitor is a mesh heuristic, not an error estimator.
Cartesian and oriented marginal monitors use the same measured field energy.
"""

from gpu import cp, quadrature
from flux import gauss_interval


def partitions(
    points,
    weights,
    n,
    bins=2048,
    smoothing=0.005,
    background=0.25,
    energy_floor=0.0,
    origins=None,
    previous_edges=None,
    phase_metric="probability",
):
    if not 0 < background <= 1 or smoothing <= 0:
        raise ValueError("A positive background and smoothing length are required")
    if points.shape[-1] != 2 or points.shape[:-1] != weights.shape:
        raise ValueError("One energy weight is required per 2D sample point")
    if energy_floor < 0:
        raise ValueError("The relative energy floor must be nonnegative")
    x, y = points[..., 0].ravel(), points[..., 1].ravel()
    weights = weights.ravel()
    origins = cp.zeros(4) if origins is None else cp.asarray(origins)
    if origins.shape != (4,):
        raise ValueError("Expected one periodic cut per monitored coordinate")
    if previous_edges is not None and previous_edges.shape != (4, n + 1):
        raise ValueError("Phase alignment requires the four previous partitions")
    if phase_metric not in ("probability", "maximum-displacement"):
        raise ValueError("Unknown periodic phase metric")
    frequency = cp.fft.rfftfreq(bins, d=1 / bins)
    smoother = cp.exp(-2 * (cp.pi * smoothing * frequency) ** 2)
    edges = []
    for axis, (coordinate, origin) in enumerate(zip((x, y, x + y, x - y), origins)):
        indices = cp.minimum(
            cp.floor(((coordinate - origin) % 1) * bins).astype(cp.int64), bins - 1
        )
        histogram = cp.bincount(indices, weights=weights, minlength=bins)
        smoothed = cp.maximum(
            cp.fft.irfft(cp.fft.rfft(histogram) * smoother, n=bins), 0
        )
        # A floor relative to the mean controls sensitivity of the fourth root
        # in low-energy regions. Zero retains the original monitor exactly.
        monitor = (smoothed + energy_floor * cp.mean(smoothed)) ** 0.25
        monitor /= cp.maximum(cp.mean(monitor), 1e-100)
        density = background + (1 - background) * monitor
        cumulative = cp.concatenate((cp.zeros(1), cp.cumsum(density)))
        cumulative /= cumulative[-1]
        levels = cp.linspace(0, 1, n + 1)
        coordinate_grid = cp.linspace(0, 1, bins + 1)
        if previous_edges is not None:
            old = previous_edges[axis, :-1] - origin
            probability = cp.floor(old) + cp.interp(
                old % 1, coordinate_grid, cumulative
            )
            # Least-squares alignment in cumulative-monitor coordinates.
            # A floating periodic phase avoids pinning one quantile at x=0.
            offset = probability - levels[:-1]
            phase = cp.mean(offset)
            if phase_metric == "maximum-displacement":
                # Quantiles increase with phase. The minimax solution balances
                # the largest positive and negative physical displacements.
                lower, upper = cp.min(offset), cp.max(offset)
                # Forty bisections resolve a unit-period phase bracket below
                # 1e-12; the positive density floor bounds inverse sensitivity.
                for _ in range(40):
                    phase = (lower + upper) / 2
                    trial = levels[:-1] + phase
                    positions = cp.floor(trial) + cp.interp(
                        trial % 1, cumulative, coordinate_grid
                    )
                    displacement = positions - old
                    positive = cp.max(displacement) + cp.min(displacement) > 0
                    lower, upper = cp.where(positive, lower, phase), cp.where(
                        positive, phase, upper
                    )
                phase = (lower + upper) / 2
            levels += phase
            edges.append(
                origin
                + cp.floor(levels)
                + cp.interp(levels % 1, cumulative, coordinate_grid)
            )
        else:
            edges.append(origin + cp.interp(levels, cumulative, coordinate_grid))
    result = cp.stack(edges)
    assert bool(cp.all(cp.diff(result, axis=1) > 0))
    return cp.ascontiguousarray(result[:2]), cp.ascontiguousarray(result[2:])


def from_fe(space, coefficients, n, **parameters):
    q = quadrature(16)
    values = cp.einsum("cfb,qb->cqf", coefficients, space.basis(q[:, :2]))
    weights = cp.sum(values**2, axis=2) * (
        2 * space.mesh.areas[:, None] * q[None, :, 2]
    )
    return partitions(space.mesh.points(q), weights, n, **parameters)


def tensor_samples(carrier, values, sample_order=None):
    """Sample a tensor polynomial with positive quadrature, entirely on GPU.

    Oversampling preserves its integrated square but resolves the spatial
    histogram more finely than its original nodal mass quadrature.
    """
    c = carrier
    order = c.np if sample_order is None else sample_order
    if order < c.np:
        raise ValueError("Sampling must integrate the squared tensor polynomial")
    nodes, weights_1d = gauss_interval(order)
    if order == c.np:
        sampled = values.reshape(c.n, c.n, values.shape[1], c.np, c.np)
    else:
        basis = cp.ones((order, c.np))
        for j in range(c.np):
            for k in range(c.np):
                if j != k:
                    basis[:, j] *= (nodes - c.nodes[k]) / (c.nodes[j] - c.nodes[k])
        sampled = cp.einsum(
            "ai,yxfij,bj->yxfab",
            basis,
            values.reshape(c.n, c.n, values.shape[1], c.np, c.np),
            basis,
        )
    coordinates = [
        c.edges[a, :-1, None] + cp.diff(c.edges[a])[:, None] * nodes for a in (0, 1)
    ]
    # Array order is (cell_y, cell_x, node_y, node_x), matching tensor mass.
    x = cp.broadcast_to(coordinates[0][None, :, None, :], (c.n, c.n, order, order))
    y = cp.broadcast_to(coordinates[1][:, None, :, None], x.shape)
    points = cp.stack((x, y), axis=-1)
    if order == c.np:
        return points, (cp.sum(values**2, axis=1) * c.mass).reshape(x.shape)
    widths = cp.diff(c.edges)
    mass = (
        widths[1, :, None, None, None]
        * widths[0, None, :, None, None]
        * weights_1d[None, None, :, None]
        * weights_1d[None, None, None, :]
    )
    return points, cp.sum(sampled**2, axis=2) * mass


def from_tensor(carrier, values, sample_order=None, **parameters):
    points, weights = tensor_samples(carrier, values, sample_order)
    return partitions(points, weights, carrier.n, **parameters)
