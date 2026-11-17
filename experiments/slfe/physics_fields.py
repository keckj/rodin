"""Render computed FE source and diffusion fields with analytic references.

Distributed under the Boost Software License, Version 1.0.
GPU solves and sampling precede host-only plotting. These repetitions supply
field data and are separate from the timing campaigns.
"""

import argparse
import json
from pathlib import Path

from matplotlib.colors import Normalize, LogNorm

from gpu import cp, Mesh, Space, quadrature
from source_study import evolve as source_evolve
from diffusion_study import evolve as diffusion_evolve
from field_report import sample, save
from report import plt, apply_style
from campaign import provenance


def validate_sampler():
    space = Space(Mesh(8, jitter=0.18, seed=817), 4)
    q = quadrature(9)
    coefficients = cp.random.RandomState(932).normal(
        size=(space.mesh.cells, 4, space.nb)
    )
    expected = cp.einsum("cfb,qb->cqf", coefficients, space.basis(q[:, :2]))
    actual = sample(space, coefficients, space.mesh.points(q), grading=0)
    error = float(cp.max(cp.abs(actual - expected)))
    assert error < 1e-10, error
    return dict(maximum_error=error, points=actual.shape[0] * actual.shape[1])


def render(output, name, space, exact, time, coefficients, records, labels, extent):
    yy, xx = cp.meshgrid(
        cp.linspace(extent[2], extent[3], 512),
        cp.linspace(extent[0], extent[1], 640),
        indexing="ij",
    )
    points = cp.stack((xx, yy), axis=-1)
    best = space.project(exact, time=time, degree=20)
    fields = sample(
        space, cp.concatenate([best, *coefficients], axis=1), points, grading=0
    )
    reference = exact(points, time)
    if reference.ndim == 3:
        reference = reference[..., 0]
    errors = cp.abs(fields - reference[..., None])
    cp.savez_compressed(
        str(output / (name + "-data.npz")),
        fields=fields,
        reference=reference,
        errors=errors,
        extent=cp.asarray(extent),
    )
    normal = Normalize(vmin=0, vmax=float(cp.max(reference)))
    maximum = float(cp.max(errors))
    error_norm = LogNorm(vmin=max(maximum * 1e-5, 1e-12), vmax=maximum)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
    for i, ax in enumerate(axes[0]):
        values = reference if i == 0 else fields[..., i]
        title = (
            "Référence"
            if i == 0
            else f"{labels[i-1]} · L2 = {max(records[i-1]['l2']):.3g}"
        )
        im = ax.imshow(
            values.get(),
            origin="lower",
            extent=extent,
            norm=normal,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(im, ax=list(axes[0]), shrink=0.8)
    for i, ax in enumerate(axes[1]):
        title = "Erreur de la projection FE" if i == 0 else f"Erreur · {labels[i-1]}"
        im = ax.imshow(
            cp.maximum(errors[..., i], error_norm.vmin).get(),
            origin="lower",
            extent=extent,
            norm=error_norm,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(im, ax=list(axes[1]), shrink=0.8, label="Erreur absolue ponctuelle")
    save(fig, output, name)
    (output / (name + ".json")).write_text(
        json.dumps(
            dict(records=records, extent=extent, provenance=provenance()), indent=2
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_style()
    (args.output / "physics-sampler-validation.json").write_text(
        json.dumps(dict(result=validate_sampler(), provenance=provenance()), indent=2)
        + "\n"
    )
    outcomes = [
        source_evolve(nfe=256, naux=n, width=0.003, monitor="fe", with_fields=True)
        for n in (32, 64)
    ]
    render(
        args.output,
        "night-source-fields",
        outcomes[0][0],
        outcomes[0][3],
        0,
        [x[2] for x in outcomes],
        [x[4] for x in outcomes],
        ["Auxiliaire 32 × 32", "Auxiliaire 64 × 64"],
        (0.68, 0.86, 0.70, 0.86),
    )
    del outcomes
    cp.get_default_memory_pool().free_all_blocks()
    outcomes = [
        diffusion_evolve(nfe=16, steps=64, method=m, with_fields=True)
        for m in ("memory", "reset")
    ]
    render(
        args.output,
        "night-diffusion-fields",
        outcomes[0][0],
        outcomes[0][3],
        outcomes[0][4]["end"],
        [x[2] for x in outcomes],
        [x[4] for x in outcomes],
        ["Avec mémoire", "Remise à zéro"],
        (0.40, 0.76, 0.36, 0.72),
    )


if __name__ == "__main__":
    main()
