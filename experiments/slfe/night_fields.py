"""Render actual FE solutions from the moving-support GPU experiment.

Distributed under the Boost Software License, Version 1.0.
All solves, characteristic integration and field sampling run on CUDA.
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
    space, initial, moving, exact, adaptive_record = evolve(
        naux=64, steps=64, with_fields=True
    )
    _, _, fixed, _, fixed_record = evolve(
        naux=64, steps=64, adaptive=False, with_fields=True
    )
    extent = (0.30, 0.57, 0.29, 0.50)
    yy, xx = cp.meshgrid(
        cp.linspace(extent[2], extent[3], 512),
        cp.linspace(extent[0], extent[1], 640),
        indexing="ij",
    )
    points = cp.stack((xx, yy), axis=-1)
    fields = sample(space, cp.concatenate((initial, fixed, moving), axis=1), points)
    reference = exact(points, 0)[..., 0]
    errors = cp.abs(fields[..., 1:] - reference[..., None])
    # The saved array is sampled numerical data, not an image approximation.
    cp.savez_compressed(
        str(args.output / "fields-data.npz"),
        fields=fields,
        reference=reference,
        errors=errors,
        extent=cp.asarray(extent),
    )
    normal = Normalize(vmin=0, vmax=float(cp.max(reference)))
    errornorm = LogNorm(vmin=1e-5, vmax=float(cp.max(errors)))
    panels = [
        (fields[..., 0], "Initial : projection FE", normal, "viridis"),
        (reference, "Référence par caractéristiques", normal, "viridis"),
        (
            fields[..., 1],
            f"Support fixe · L2 = {fixed_record['l2'][0]:.3g}",
            normal,
            "viridis",
        ),
        (
            fields[..., 2],
            f"Support mobile · L2 = {adaptive_record['l2'][0]:.3g}",
            normal,
            "viridis",
        ),
        (
            cp.maximum(errors[..., 0], 1e-5),
            "Erreur absolue · support fixe",
            errornorm,
            "magma",
        ),
        (
            cp.maximum(errors[..., 1], 1e-5),
            "Erreur absolue · support mobile",
            errornorm,
            "magma",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), layout="constrained")
    for ax, (values, title, norm, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(
            values.get(),
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        "Transport cellulaire jusqu'à t = 0,25\n32768 triangles P4 · auxiliaire 64 × 64 Q4 · 64 pas",
        fontsize=15,
    )
    save(fig, args.output, "night-fields")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for record, label in ((fixed_record, "Fixe"), (adaptive_record, "Mobile")):
        history = record["history"]
        times = [row["step"] * record["end"] / record["steps"] for row in history]
        axes[0].plot(
            [row["detected"][0] for row in history],
            [row["detected"][1] for row in history],
            ".-",
            label=label,
        )
        axes[1].plot(
            times, [1 - row["energy_ratio"] for row in history], ".-", label=label
        )
    centers = [row["center"] for row in adaptive_record["history"] if row["restarted"]]
    axes[0].scatter(
        [c[0] for c in centers],
        [c[1] for c in centers],
        marker="x",
        s=80,
        color="black",
        label="Changement de support",
    )
    axes[0].set(
        xlabel="Centre d'énergie : x",
        ylabel="Centre d'énergie : y",
        title="Décisions prises à partir du champ calculé",
    )
    axes[1].set(
        xlabel="Temps physique",
        ylabel="Perte relative d'énergie",
        title="Dissipation totale mesurée",
    )
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    save(fig, args.output, "night-trajectory")
    (args.output / "night-fields.json").write_text(
        json.dumps(
            dict(fixed=fixed_record, moving=adaptive_record, provenance=provenance()),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
