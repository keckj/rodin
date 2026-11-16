"""Actual GPU field plots for several separated transported profiles.

Distributed under the Boost Software License, Version 1.0.
Field panels show the sum; error panels show the maximum over the four fields.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from moving_study import evolve
from field_report import sample, save
from report import plt, apply_style
from matplotlib.colors import Normalize, LogNorm
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_style()
    parameters = dict(
        nfe=128, naux=64, steps=128, fields=4, spread=0.15, with_fields=True
    )
    space, initial, centroid, exact, record_c = evolve(**parameters)
    _, _, energy, _, record_e = evolve(**parameters, mesh_policy="energy")
    extent = (0.0, 0.85, 0.0, 0.7)
    yy, xx = cp.meshgrid(
        cp.linspace(extent[2], extent[3], 640),
        cp.linspace(extent[0], extent[1], 768),
        indexing="ij",
    )
    points = cp.stack((xx, yy), axis=-1)
    sampled = sample(space, cp.concatenate((initial, centroid, energy), axis=1), points)
    reference = exact(points, 0)
    error_c = cp.max(cp.abs(sampled[..., 4:8] - reference), axis=-1)
    error_e = cp.max(cp.abs(sampled[..., 8:] - reference), axis=-1)
    sums = [cp.sum(sampled[..., 4 * i : 4 * (i + 1)], axis=-1) for i in range(3)]
    normal = Normalize(
        vmin=min(0, *(float(x.min()) for x in sums)),
        vmax=max(
            float(cp.sum(reference, axis=-1).max()), *(float(x.max()) for x in sums)
        ),
    )
    errornorm = LogNorm(vmin=1e-5, vmax=max(float(error_c.max()), float(error_e.max())))
    panels = [
        (sums[0], "Initial : quatre profils", normal, "viridis"),
        (cp.sum(reference, axis=-1), "Référence : quatre profils", normal, "viridis"),
        (
            sums[1],
            f"Centre unique · L2 max = {max(record_c['l2']):.3g}",
            normal,
            "viridis",
        ),
        (
            sums[2],
            f"Équidistribution · L2 max = {max(record_e['l2']):.3g}",
            normal,
            "viridis",
        ),
        (
            cp.maximum(error_c, 1e-5),
            "Erreur maximale · centre unique",
            errornorm,
            "magma",
        ),
        (
            cp.maximum(error_e, 1e-5),
            "Erreur maximale · équidistribution",
            errornorm,
            "magma",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), layout="constrained")
    for ax, (value, title, norm, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(
            value.get(),
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        "Structures séparées : adaptation des quatre directions\n32768 triangles P4 · auxiliaire 64 × 64 Q4 · t = 0,25",
        fontsize=15,
    )
    save(fig, args.output, "night-separated-fields")
    (args.output / "night-separated-fields.json").write_text(
        json.dumps(
            dict(centroid=record_c, energy=record_e, provenance=provenance()), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
