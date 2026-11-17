"""Scientific maps of complete-state reaction, sampled entirely on CUDA.

Distributed under the Boost Software License, Version 1.0.
Matplotlib renders saved arrays; normalization is by the reference peak.
"""

import numpy as np
from matplotlib.colors import LogNorm, Normalize

from gpu import cp
from field_report import sample, save
from report import plt, apply_style


def render_gpu(output, space, exact, values, nfe, steps, strength):
    methods = ("fe-only", "coercive-lift", "ambient-moments")
    if not all(method in values for method in methods):
        raise ValueError("Field comparison requires all three reaction models")
    output.mkdir(exist_ok=True, parents=True)
    extent = (0.28, 0.58, 0.28, 0.55)
    yy, xx = cp.meshgrid(
        cp.linspace(extent[2], extent[3], 512),
        cp.linspace(extent[0], extent[1], 640),
        indexing="ij",
    )
    points = cp.stack((xx, yy), axis=-1)
    reference = exact(points)[..., 0]
    fields = sample(space, cp.concatenate([values[m] for m in methods], axis=1), points)
    peak = cp.max(reference)
    fields, reference = fields / peak, reference / peak
    errors = cp.abs(fields - reference[..., None])
    row = int(cp.argmax(cp.max(reference, axis=1)))
    name = f"night-reaction-fields-{nfe}-{steps}-{strength:g}"
    path = output / f"{name}-data.npz"
    cp.savez_compressed(
        str(path),
        fields=fields,
        reference=reference,
        errors=errors,
        extent=cp.asarray(extent),
        peak=peak,
        row=cp.asarray(row),
    )
    render(path)


def render(path):
    data = np.load(path)
    fields, reference, errors = data["fields"], data["reference"], data["errors"]
    extent, row = data["extent"].tolist(), int(data["row"])
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    normal = Normalize(0, 1)
    for ax, field, title in zip(
        axes[0, :2],
        (reference, fields[..., 2]),
        ("Référence par caractéristiques", "Moments du champ complet"),
    ):
        artist = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            norm=normal,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(
        artist, ax=list(axes[0, :2]), shrink=0.8, label="Champ / maximum exact"
    )
    labels = ("FE seule", "Amortissement uniforme", "Moments complets")
    ax = axes[0, 2]
    x = np.linspace(extent[0], extent[1], fields.shape[1])
    ax.plot(x, reference[row], "k--", label="Référence", linewidth=2)
    for i, label in enumerate(labels):
        ax.plot(x, fields[row, :, i], label=label, alpha=0.8)
    ax.set_yscale("symlog", linthresh=1)
    y = np.linspace(extent[2], extent[3], fields.shape[0])[row]
    ax.set(xlabel="x", ylabel="Champ / maximum exact", title=f"Coupe à y = {y:.3f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    norm = LogNorm(1e-6, max(1.0, float(errors.max())))
    for i, (ax, label) in enumerate(zip(axes[1], labels)):
        artist = ax.imshow(
            np.maximum(errors[..., i], norm.vmin),
            origin="lower",
            extent=extent,
            norm=norm,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set(title=f"Erreur · {label}", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(
        artist, ax=list(axes[1]), shrink=0.8, label="Erreur absolue / maximum exact"
    )
    save(fig, path.parent, path.name.removesuffix("-data.npz"))
    plt.close(fig)
