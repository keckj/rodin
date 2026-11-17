"""Evidence for target-moment retention and unresolved reaction closure.

Distributed under the Boost Software License, Version 1.0.
Host-only report generation from GPU measurements.
"""

import json

from report import plt


def append_target_seed(r, raw, source):
    rows = [x["result"] for x in raw.get("unresolved-reaction", [])]
    rows += [
        x["result"]
        for x in raw.get("unresolved-target", [])
        if x["result"]["method"] == "target"
    ]
    if not rows:
        return
    r.section("Retenir le moment cible donne une garantie d’approximation plus forte")
    r.paragraph(
        "L’espace précédent part de la mémoire entrante d, puis du moment caché h = P_W f − R P_V f et de R R* h. Il peut manquer une direction pourtant immédiatement disponible : le moment brut m = P_W f. La nouvelle option conserve d et m dans le même espace partagé. Elle n’ajoute aucun calcul de moments et utilise deux directions par composante au lieu de trois dans cette expérience."
    )
    r.paragraph(
        "Si m appartient à l’espace de directions X, alors m appartient à V + (I − P_V)X. La projection orthogonale du champ cible f sur cet espace a donc une erreur au plus égale à celle de P_W f. Si f appartient déjà à W, la compression le conserve exactement. La mémoire entrante reste incluse pour conserver l’identité à physique nulle. Cette garantie suppose que les directions nécessaires survivent à la décision numérique de rang; le seuil de Gram peut la perturber près d’une dépendance linéaire."
    )
    r.tex.append(r"\[\|f-\mathcal P_X f\|\leq\|f-P_Wf\|,\qquad P_Wf\in X.\]")
    r.paragraph(
        "Le test est une réaction sans transport, partant du champ constant un, avec r(x) = 20(1 + 0,95 sin(2πx)) jusqu’à T = 1. Elle crée une structure sous-maille malgré une condition initiale FE exacte. La référence comparée est l’observation P_V exp(−r), intégrée avec deux quadratures raffinées. L’erreur ci-dessous est la distance à cette observation FE exacte, normalisée par sa norme; elle ne mesure pas l’erreur du champ continu oscillant représenté seulement en FE."
    )
    r.paragraph(
        "Le comparateur exponentiel diagonalise exactement la matrice de réaction Galerkin grossière sur GPU. Diminuer le pas de la reconstruction FE converge vers cette dynamique fermée, qui diffère de P_V exp(−r). Une évolution directe sur le support auxiliaire fin, avec observation FE à chaque pas, est aussi incluse. Elle est ici le comparateur le plus efficace : aucune opération FE supplémentaire n’impose la compression."
    )
    labels = {
        "full": "Ancienne base (3 directions)",
        "target": "Moment cible (2 directions)",
        "reset": "Reconstruction FE",
        "auxiliary": "Auxiliaire direct",
        "fe-exponential": "Exponentielle FE",
    }
    colors = {
        "full": "#d78337",
        "target": "#8158a3",
        "reset": "#b14245",
        "auxiliary": "#3b8b76",
        "fe-exponential": "#444444",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (degree, n) in zip(axes, ((1, 4), (4, 8))):
        for method in ("full", "target", "reset", "auxiliary"):
            selected = sorted(
                [
                    x
                    for x in rows
                    if x["degree"] == degree and x["nfe"] == n and x["method"] == method
                ],
                key=lambda x: x["steps"],
            )
            if selected:
                ax.loglog(
                    [x["steps"] for x in selected],
                    [max(x["relative_observation_error"], 1e-16) for x in selected],
                    "o-",
                    label=labels[method],
                    color=colors[method],
                    ms=4,
                )
        exact = next(
            x
            for x in rows
            if x["degree"] == degree
            and x["nfe"] == n
            and x["method"] == "fe-exponential"
        )
        ax.axhline(
            exact["relative_observation_error"],
            ls=":",
            color=colors["fe-exponential"],
            label=labels["fe-exponential"],
        )
        ax.set(
            xlabel="Nombre de pas de réaction",
            ylabel="Erreur relative de l’observation FE",
            title=f"P{degree}, n FE = {n}, n auxiliaire = 128",
        )
        ax.grid(True, which="both", alpha=0.18)
    axes[0].legend(fontsize=7, loc="best")
    r.save_figure(
        fig,
        "night-target-seed",
        "Conserver le moment auxiliaire brut évite l’essentiel de la perte de compression dans ce problème sous-résolu. La reconstruction FE tend vers une limite de fermeture différente. Les résultats à un seul pas ne testent pas l’accumulation : tous les modèles à moments FE exacts donnent alors directement la bonne observation.",
    )
    selected = [
        x for x in rows if x["degree"] == 1 and x["nfe"] == 4 and x["steps"] == 64
    ]
    r.table(
        ["Méthode", "Erreur observation", "Erreur masse", "Évolution (s)"],
        [
            [
                labels[x["method"]],
                x["relative_observation_error"],
                x["relative_mass_error"],
                x["evolution_seconds"],
            ]
            for x in selected
        ],
    )
    r.paragraph(
        "Les temps sont des mesures uniques après préparation. Le support auxiliaire est Q4 et reste fixe; les degrés FE un, deux et quatre et deux résolutions grossières sont testés. La nouvelle option améliore fortement ce test, mais elle ne remplace pas automatiquement les directions précédentes : sur l’advection-réaction à neuf modes, les erreurs sont proches, parfois légèrement moins bonnes. La comparaison avec l’auxiliaire direct empêche de conclure à un avantage de coût pour cette réaction seule."
    )
    validation = source / "target-seed.validation.json"
    if validation.exists():
        data = json.loads(validation.read_text())["controls"]
        exact = [
            x["exact_auxiliary_target_relative_error"]
            for x in data
            if "exact_auxiliary_target_relative_error" in x
        ]
        r.paragraph(
            f"Quinze contrôles GPU indépendants vérifient la conservation des observations, la contraction, la domination de l’énergie projetée par rapport au seul moment W et la reproduction des cibles appartenant à W. Ils couvrent réactions linéaire et cubique, espaces séparés ou partagés et degrés 1, 2, 4. La plus grande erreur relative physique de reproduction d’une cible W vaut {max(exact):.3g}, mesurée par quadrature positive. Les quadratures de référence de l’expérience sous-résolue diffèrent au plus de {max(x['reference_quadrature_error'] for x in rows):.3g} en norme FE relative."
        )
    r.paragraph(
        "La propriété est celle d’une projection sur un espace enrichi qui contient une approximation connue de sa cible. Elle ne constitue pas à elle seule une nouveauté bibliographique. Sa conséquence algorithmique concrète est de choisir une direction disponible qui assure cette propriété, plutôt que d’espérer la récupérer par quelques puissances de R R*. Le cas P4, n = 8 atteint seulement environ 2e−8 avec cette réalisation : le conditionnement et la troncature de Gram restent à contrôler lorsque la cible est presque dans V."
    )
    rank_file = source / "target-seed-rank.json"
    if rank_file.exists():
        rank_data = json.loads(rank_file.read_text())
        r.paragraph(
            "Le diagnostic instrumenté P4, n = 8 isole cette limite. Au deuxième pas, une valeur propre de Gram voisine de 3,1e−13 passe sous le seuil 1e−12 : le rang conservé devient un. Une projection indépendante utilisant exactement les mêmes directions ne reproduit alors plus leur moment cible, avec une erreur physique relative de 1,24e−7. L’observation FE reste préservée à ce pas, mais l’erreur apparaît dans les étapes suivantes. Le problème de rang est donc mesuré, et la garantie de meilleure approximation reste conditionnelle à la rétention effective du moment. Ces calculs instrumentés ne figurent pas dans les chronométrages."
        )
        r.table(
            [
                "Pas",
                "Rang",
                "Petite valeur propre",
                "Défaut cible W",
                "Erreur observation",
            ],
            [
                [
                    x["step"],
                    x["retained_rank"],
                    min(x["gram_eigenvalues"]),
                    x["physical_relative_target_retention_error"],
                    x["relative_observation_error"],
                ]
                for x in rank_data["controls"]
            ],
        )
    refined = [
        x["result"]
        for x in raw.get("unresolved-target-quadrature", [])
        if x["result"]["method"] != "fe-exponential"
    ]
    differences = []
    for x in refined:
        y = next(
            (
                y
                for y in rows
                if all(x[k] == y[k] for k in ("degree", "nfe", "steps", "method"))
            ),
            None,
        )
        if y is not None:
            differences.append(
                (
                    x["steps"],
                    abs(
                        x["relative_observation_error"]
                        - y["relative_observation_error"]
                    ),
                )
            )
    if differences:
        r.paragraph(
            f"Doubler la subdivision de la quadrature pondérée modifie l’erreur relative d’observation d’au plus {max(d for _,d in differences):.3g}; pour les trajectoires de 64 pas, l’écart maximal vaut {max(d for steps,d in differences if steps==64):.3g}. Le plus grand écart concerne un unique grand pas de réaction P4 sur le maillage le plus grossier."
        )
    evidence = [
        r"\paragraph{A raw target moment repairs an avoidable compression loss.}",
        r"A reaction-only diagnostic starts from one with rate $20(1+0.95\sin(2\pi x))$ and final time one. It creates unresolved structure on a coarse FE mesh. The measured quantity is the error of the FE observation against $P_V\exp(-r)$, not the continuous-field approximation error. The exact coarse Galerkin matrix exponential provides a separate closure comparator.",
    ]
    if selected:
        evidence += [
            r"\begin{center}\small\begin{tabular}{lrr}\toprule Method & Observation error & Evolution (s)\\\midrule"
        ]
        english = {
            "full": "Old three-direction space",
            "target": "Raw target, two directions",
            "reset": "FE reconstruction",
            "auxiliary": "Direct auxiliary evolution",
        }
        for x in selected:
            evidence.append(
                f"{english[x['method']]} & {x['relative_observation_error']:.3e} & {x['evolution_seconds']:.4f}"
                + r"\\"
            )
        evidence += [
            r"\bottomrule\end{tabular}\end{center}",
            r"These are single evolution measurements for $P_1$, $n_V=4$, $Q_4$, $n_W=128$ and 64 steps, excluding preparation. The direct auxiliary solver remains faster and requires no compression in this reaction-only example. Thus the experiment motivates the retained direction but establishes no overall efficiency advantage.",
        ]
    evidence += [
        r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{night-target-seed.pdf}",
        r"\caption{Coarse FE observation of an unresolved reaction. Retaining the raw auxiliary target moment substantially reduces compression error. The numerical rank cutoff remains relevant near dependent directions.}\end{figure}",
    ]
    (r.output / "target_seed_evidence.tex").write_text("\n".join(evidence) + "\n")
