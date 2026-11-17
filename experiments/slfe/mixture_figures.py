"""Render recorded cell RMS values on a complete periodic display window.

Distributed under the Boost Software License, Version 1.0.
Host-only plotting; all FE coefficients and RMS reductions originate on CUDA.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.colors import LogNorm
from report import plt, apply_style


def render(path):
    data = np.load(path)
    triangles, values = data["triangles"], data["closure_rms"]
    # FE triangles use an unwrapped periodic coordinate chart. Replicate
    # intersecting cells for display so the unit torus has no artificial gaps.
    copies, cell_values = [], []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            shifted = triangles + np.asarray([x, y])
            intersects = np.all(shifted.max(axis=1) >= 0, axis=1)
            intersects &= np.all(shifted.min(axis=1) <= 1, axis=1)
            copies.append(shifted[intersects])
            cell_values.append(values[:, intersects])
    displayed = np.concatenate(copies)
    displayed_values = np.concatenate(cell_values, axis=1)
    triangulation = Triangulation(
        displayed[..., 0].ravel(),
        displayed[..., 1].ravel(),
        np.arange(3 * len(displayed)).reshape(-1, 3),
    )
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    maximum = max(1e-13, float(values.max()))
    labels = ("Chebyshev", "Galerkin séparé", "Galerkin partagé")
    for ax, label, field in zip(axes, labels, displayed_values):
        artist = ax.tripcolor(
            triangulation,
            facecolors=np.maximum(field, 1e-16),
            norm=LogNorm(1e-14, maximum),
            cmap="magma",
            shading="flat",
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            aspect="equal",
            title=label,
            xlabel="x",
            ylabel="y",
        )
    fig.colorbar(
        artist, ax=axes, label=r"$\|\sum_f Y_f-1\|_{L^2(K)}/\sqrt{|K|}$", shrink=0.85
    )
    name = path.name.removesuffix("-data.npz")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(path.parent / f"{name}.{suffix}", dpi=160, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    for path in args.input.glob("night-mixture-closure-*-data.npz"):
        render(path)
