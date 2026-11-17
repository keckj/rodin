"""Plots and evidence for the CUDA Hamiltonian chart experiments.

Distributed under the Boost Software License, Version 1.0.
Only recorded results are processed on the host.
"""

import json
import math
import statistics

from report import plt


SCHEMES = {
    "strang": "Strang",
    "triple4": "Composition 4 (3 stages)",
    "suzuki4": "Composition 4 (5 stages)",
    "yoshida6": "Yoshida 6 (7 stages)",
}
COLORS = {
    "strang": "#597b90",
    "triple4": "#d78337",
    "suzuki4": "#3b8b76",
    "yoshida6": "#8158a3",
}


def append_hamiltonian(r, raw, source):
    benchmarks = []
    prefix = (
        "hamiltonian-compact-benchmark"
        if raw.get("hamiltonian-compact-benchmark")
        else "hamiltonian-benchmark"
    )
    for name in (
        prefix,
        prefix + "-triple4",
        prefix + "-suzuki4",
        prefix + "-yoshida6",
    ):
        for record in raw.get(name, []):
            value = dict(record["result"])
            value.setdefault(
                "scheme", "rk4" if value["method"].startswith("dg-") else "strang"
            )
            benchmarks.append(value)
    for name in (
        "hamiltonian-dg-refinement",
        "hamiltonian-dg-central-fine",
        "hamiltonian-dg-upwind-fine",
    ):
        benchmarks.extend(x["result"] for x in raw.get(name, []))
    studies = [
        x["result"]
        for name in ("hamiltonian", "hamiltonian-fine")
        for x in raw.get(name, [])
    ]
    if not studies:
        return
    r.section("Vitesses FE filtrées : cisaillements dans des coordonnées entières")
    r.paragraph(
        "Une fonction de courant P4 sur triangles perturbés fournit ses moments de Fourier par quadrature GPU. Chaque direction entière primitive devient une coordonnée d’une carte de déterminant un. Les rotations du tenseur sont des permutations exactes; les cisaillements de coordonnées utilisent les mêmes projections conservatives que le transport. Les harmoniques colinéaires partagent une carte. Le support auxiliaire est uniforme : cette extension ne couvre pas encore les supports adaptés non uniformes ni les frontières physiques."
    )
    r.paragraph(
        "Si C est la contraction réalisant la carte et S le cisaillement, l’opérateur implémenté est K = I + C*(S − I)C. Il est exactement l’identité à durée nulle. Avec D = (I − C*C)^(1/2), l’application L = (C,D) est isométrique et K = L* diag(S,I)L : K est donc contractant. Les constantes et la masse sont préservées. D sert uniquement à la preuve et n’est jamais calculé. Les compositions de ces opérateurs héritent de la contraction, y compris pour des durées signées."
    )
    r.paragraph(
        "Le découpage d’un Hamiltonien trigonométrique en cisaillements est classique : Quispel–McLaren (2003) et Zanna (2015). Les compositions d’ordre supérieur sont elles aussi classiques, notamment Yoshida (1990). Les projections spatiales détruisent la réversibilité exacte; les ordres quatre et six indiquent la composition continue et sont vérifiés expérimentalement avant le plateau spatial. Aucune preuve d’ordre temporel uniforme à maillage fixé n’est déduite de la stabilité."
    )
    r.paragraph(
        "Le premier problème transporte deux bosses périodiques jusqu’à T = 0,5, avec neuf directions allant jusqu’à (3,2), sur 2 n² triangles P4 perturbés. La grille auxiliaire comporte (2n)² cellules Q4. La référence utilise des caractéristiques RK4 indépendantes de la décomposition. Son raffinement de 512 à 1024 sous-pas change les pieds de moins de 5,2e−14 sur 4096 points. La vitesse initiale est ici à spectre fini : l’erreur du filtrage FE est mesurée séparément et devient négligeable au raffinement."
    )
    validation = source / "hamiltonian-composition.validation.json"
    if validation.exists():
        control = json.loads(validation.read_text())
        norms = [
            x["dense_transport_norm"]
            for x in control["controls"]
            if "dense_transport_norm" in x
        ]
        errors = [
            v
            for x in control["controls"]
            for k, v in x.items()
            if k
            in (
                "mass_error",
                "constant_error",
                "zero_time_identity_error",
                "adjoint_error",
            )
        ]
        r.paragraph(
            f"Les contrôles GPU comprennent {len(control['controls'])} enregistrements algébriques aux degrés 1, 2 et 4 et {len(control['characteristics'])} comparaisons avec des trajectoires monomodes exactes. La plus grande norme singulière mesurée des transports complets est {max(norms):.14g}; le plus grand défaut de masse, constante, identité à temps nul ou adjoint est {max(errors):.3g}. Ces contrôles incluent les trois compositions à sous-pas négatifs."
        )
    if benchmarks:
        accepted = [x for x in benchmarks if x["stable_probe"]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, n in zip(axes, (64, 128)):
            for scheme, label in SCHEMES.items():
                rows = sorted(
                    [
                        x
                        for x in accepted
                        if x["nfe"] == n
                        and x["method"] == "memory"
                        and x["scheme"] == scheme
                    ],
                    key=lambda x: x["steps"],
                )
                if rows:
                    ax.loglog(
                        [x["steps"] for x in rows],
                        [max(x["relative_l2"]) for x in rows],
                        "o-",
                        label=label,
                        color=COLORS[scheme],
                        ms=4,
                    )
            floor = max(next(x["relative_fe_floor"] for x in accepted if x["nfe"] == n))
            ax.axhline(floor, color="#555555", ls=":", label="Projection FE")
            ax.set(
                xlabel="Nombre de pas sur T = 0,5",
                ylabel="Erreur relative L2 (max. des 2 champs)",
                title=f"n FE = {n}, n auxiliaire = {2*n}",
            )
            ax.grid(True, which="both", alpha=0.18)
        axes[0].legend(fontsize=8)
        r.save_figure(
            fig,
            "night-hamiltonian-order",
            "Raffinement temporel, même maillage et même vitesse filtrée. Les compositions d’ordre supérieur réduisent le nombre de pas nécessaire; leur erreur atteint ensuite un plateau spatial. Toutes les courbes conservent la mémoire physique.",
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, n in zip(axes, (64, 128)):
            for scheme, label in SCHEMES.items():
                for method, style in (("memory", "-"), ("reset", "--")):
                    rows = sorted(
                        [
                            x
                            for x in accepted
                            if x["nfe"] == n
                            and x["method"] == method
                            and x["scheme"] == scheme
                        ],
                        key=lambda x: x["evolution_seconds"],
                    )
                    if rows:
                        ax.loglog(
                            [x["evolution_seconds"] for x in rows],
                            [max(x["relative_l2"]) for x in rows],
                            "o" + style,
                            color=COLORS[scheme],
                            ms=4,
                            label=label if method == "memory" else None,
                        )
            for method, color, marker in (
                ("dg-central", "#333333", "s"),
                ("dg-upwind", "#b14245", "^"),
            ):
                rows = sorted(
                    [x for x in accepted if x["nfe"] == n and x["method"] == method],
                    key=lambda x: x["evolution_seconds"],
                )
                ax.loglog(
                    [x["evolution_seconds"] for x in rows],
                    [max(x["relative_l2"]) for x in rows],
                    marker + "-",
                    color=color,
                    label=method + " / RK4",
                    ms=5,
                )
            ax.set(
                xlabel="Évolution GPU capturée (s, médiane de 3)",
                ylabel="Erreur relative L2 (max. des 2 champs)",
                title=f"n FE = {n}",
            )
            ax.grid(True, which="both", alpha=0.18)
        axes[0].legend(fontsize=7)
        r.save_figure(
            fig,
            "night-hamiltonian-cost",
            "Coût contre précision à espaces FE et vitesse identiques. Traits pleins : mémoire; pointillés : reconstruction FE à chaque pas. DG central et upwind utilisent RK4. Les étapes des deux familles sont capturées dans des graphes CUDA; le champ FE est actualisé à chaque étape SL. Les assemblages et la préparation sont exclus de ces temps.",
        )
        r.paragraph(
            "Le comparateur DG utilise exactement le même espace P4 et les mêmes coefficients de vitesse filtrée. Quatre tailles de pas RK4 sont testées; celles dont la sonde aléatoire amplifie l’énergie sont rejetées avant chronométrage. Ce contrôle empirique ne constitue pas une preuve de stabilité RK4. Les répétitions sont successives par configuration, sans entrelacement des méthodes; leur dispersion est conservée dans les données. Une petite différence de temps ne suffit donc pas à conclure à un gain robuste."
        )
        rows = []
        for n, tolerance in ((64, 1e-6), (128, 1e-7), (128, 3e-8)):
            for family in ("dg", "memory", "reset"):
                candidates = [
                    x
                    for x in accepted
                    if x["nfe"] == n
                    and max(x["relative_l2"]) <= tolerance
                    and (
                        x["method"].startswith("dg-")
                        if family == "dg"
                        else x["method"] == family
                    )
                ]
                if candidates:
                    best = min(candidates, key=lambda x: x["evolution_seconds"])
                    rows.append(
                        [
                            n,
                            tolerance,
                            best["method"] + "/" + best["scheme"],
                            best["steps"],
                            max(best["relative_l2"]),
                            best["evolution_seconds"],
                        ]
                    )
                else:
                    rows.append([n, tolerance, family, "—", "non atteint", "—"])
        r.table(
            ["n FE", "Tolérance", "Meilleur testé", "Pas", "Erreur", "Évolution (s)"],
            rows,
        )
        r.paragraph(
            f"{len(benchmarks)} configurations capturées sont enregistrées, dont {len(benchmarks)-len(accepted)} rejetées par le contrôle d’énergie. Les résultats portent sur une advection prescrite avec observation FE intermédiaire; aucune physique FE supplémentaire n’est appliquée dans ce comparatif. Ils ne démontrent pas que la mémoire soit indispensable à une EDP couplée."
        )
    storage = source / "tensor-storage.validation.json"
    if storage.exists():
        controls = json.loads(storage.read_text())["controls"]
        r.paragraph(
            "La première tentative au rayon spectral K=6 a saturé la mémoire GPU : 31,6 Go étaient déjà alloués. "
            "Le stockage a été corrigé en partageant les cartes indépendantes du sous-pas, puis les blocs d’intersection "
            "identiques entre cellules longitudinales d’une grille uniforme. Les partitions non uniformes et les changements "
            "de partition conservent leur représentation complète. "
            f"Les {len(controls)} contrôles contre cette représentation donnent un écart maximal de "
            f"{max(x.get('max_difference', 0) for x in controls):.3g} sur des champs aléatoires. "
            "La réduction des blocs uniformes est proportionnelle au nombre de cellules par direction; "
            "le filtrage, la quadrature et l’opérateur mathématique sont identiques."
        )
    orders = []
    for scheme in ("triple4", "suzuki4", "yoshida6"):
        rows = {
            x["steps"]: x
            for x in benchmarks
            if x["nfe"] == 128
            and x["method"] == "memory"
            and x["scheme"] == scheme
            and x["stable_probe"]
        }
        if 2 in rows and 4 in rows:
            order = math.log(
                max(rows[2]["relative_l2"]) / max(rows[4]["relative_l2"]), 2
            )
            orders.append(
                [
                    SCHEMES[scheme],
                    max(rows[2]["relative_l2"]),
                    max(rows[4]["relative_l2"]),
                    order,
                ]
            )
    if orders:
        r.table(
            ["Composition", "Erreur à 2 pas", "Erreur à 4 pas", "Ordre apparent"],
            orders,
        )
    matched_prefix = (
        "hamiltonian-tuned-matched"
        if raw.get("hamiltonian-tuned-matched-0")
        else "hamiltonian-compact-matched"
    )
    matched = [
        x["result"] for i in range(5) for x in raw.get(f"{matched_prefix}-{i}", [])
    ]
    matched_summary = []
    if matched:
        r.section("Répétitions alternées du comparatif à neuf modes")
        for method, steps in sorted({(x["method"], x["steps"]) for x in matched}):
            rows = [
                x
                for x in matched
                if x["method"] == method and x["steps"] == steps and x["stable_probe"]
            ]
            if not rows:
                continue
            times = [x["evolution_seconds"] for x in rows]
            matched_summary.append(
                dict(
                    method=method,
                    steps=steps,
                    repetitions=len(rows),
                    median_seconds=statistics.median(times),
                    minimum_seconds=min(times),
                    maximum_seconds=max(times),
                    relative_error=max(max(x["relative_l2"]) for x in rows),
                )
            )
        labels = {
            "dg-central": "DG central",
            "dg-upwind": "DG upwind",
            "memory": "Mémoire / Y6",
            "reset": "Reset / Y6",
        }
        r.table(
            ["Méthode", "Pas", "Erreur max.", "Médiane (s)", "Étendue (s)", "Rép."],
            [
                [
                    labels[x["method"]],
                    x["steps"],
                    x["relative_error"],
                    x["median_seconds"],
                    f"{x['minimum_seconds']:.3g}–{x['maximum_seconds']:.3g}",
                    x["repetitions"],
                ]
                for x in matched_summary
            ],
        )
        r.paragraph(
            "Les blocs alternent l’ordre des six configurations pour limiter le biais de chauffe. Le nombre de répétitions disponibles figure dans la dernière colonne. Chaque bloc reconstruit les opérateurs et réchauffe les graphes; les temps ci-dessus mesurent exclusivement l’évolution. Le maillage FE, les deux champs, la vitesse et T = 0,5 sont identiques. Les étapes DG utilisent les échelles choisies pour chaque flux; un raffinement jusqu’à un quart d’échelle recherche le plus grand pas passant la sonde d’énergie. Le protocole et les rejets sont conservés. Le gain éventuel à une tolérance donnée doit être attribué à la composition de transport; une reconstruction sans mémoire presque aussi efficace empêche de l’attribuer à la mémoire persistante."
        )
    (r.output / "hamiltonian_summary.json").write_text(
        json.dumps(dict(matched=matched_summary, orders=orders), indent=2) + "\n"
    )
    generator = source / "composition-generator.validation.json"
    if generator.exists():
        controls = json.loads(generator.read_text())["controls"]
        r.section("Pourquoi réduire encore le pas finit par ajouter de la dissipation")
        r.paragraph(
            "Pour une translation projetée sur un espace DG fixé, S(−t)=S(t)*. Sa dérivée à droite s’écrit A−D, avec A antisymétrique et D positif. Au premier ordre, S(t)=I+t A−|t|D. Une composition de poids de somme un a donc pour générateur A−αD, où α est la somme des valeurs absolues des poids. Les sous-pas négatifs augmentent la dissipation des sauts. La stabilité de la composition n’implique pas un ordre quatre ou six uniforme à espace discret fixé; les ordres observés sur des champs lisses précèdent un plateau spatial."
        )
        rows = [x for x in controls if x["degree"] == 4 and x["duration"] == 1e-8]
        r.table(
            ["Composition", "Facteur prédit", "Facteur mesuré", "Défaut du générateur"],
            [
                [
                    SCHEMES[x["scheme"]],
                    x["absolute_weight_sum"],
                    x["measured_dissipation_multiplier"],
                    x["generator_relative_residual"],
                ]
                for x in rows
            ],
        )
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for scheme, label in SCHEMES.items():
            selected = sorted(
                [x for x in controls if x["degree"] == 4 and x["scheme"] == scheme],
                key=lambda x: x["duration"],
            )
            if selected:
                ax.loglog(
                    [x["duration"] for x in selected],
                    [x["generator_relative_residual"] for x in selected],
                    "o-",
                    label=label,
                    color=COLORS[scheme],
                )
        ax.set(
            xlabel="Durée positive de la composition",
            ylabel="Défaut relatif du générateur prédit",
            title="Matrices denses GPU, translation Q4 sur une grille fixe",
        )
        ax.grid(True, which="both", alpha=0.18)
        ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-composition-generator",
            "Différences finies unilatérales de matrices GPU indépendantes. Le défaut décroissant confirme le générateur avec dissipation amplifiée; ce contrôle évite d’interpréter un ordre apparent avant plateau comme un théorème d’ordre à maillage fixé.",
        )
    images = sorted(r.output.glob("night-hamiltonian-fields-*.pdf"))
    initial = r.output / "night-hamiltonian-fields-yoshida6-128-8.pdf"
    if not initial.exists():
        initial = next(
            (p for p in images if "broad" not in p.name and "reaction" not in p.name),
            None,
        )
    if initial:
        r.figure(
            initial.stem,
            "Champs transportés et erreurs ponctuelles contre les caractéristiques indépendantes. Les valeurs négatives éventuelles ne sont pas exclues par la contraction L2.",
        )
    broad = [
        x["result"]
        for name in (
            "hamiltonian-broad",
            "hamiltonian-broad-fine",
            "hamiltonian-broad-tail",
        )
        for x in raw.get(name, [])
    ]
    if broad:
        r.section("Spectre non limité : isoler l’erreur de vitesse")
        r.paragraph(
            "La seconde fonction de courant est 0,04 exp(0,7 cos(2π(x−0,5)) + 0,45 cos(2π(y−0,5)) − 1,15). Ses moments proviennent de sa projection FE, et seuls les modes du carré de rayon K sont retenus. La référence intègre directement la vitesse analytique non tronquée. Une FFT GPU indépendante de la fonction analytique contrôle les moments FE retenus; elle ne sert pas à construire le transport. Le nombre de modes croît comme 2K(K+1), ce qui augmente directement le coût des cisaillements."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for n in sorted({x["nfe"] for x in broad}):
            rows = sorted(
                [
                    x
                    for x in broad
                    if x["nfe"] == n and x["method"] == "memory" and x["steps"] == 32
                ],
                key=lambda x: x["band"],
            )
            if rows:
                axes[0].semilogy(
                    [x["band"] for x in rows],
                    [x["velocity_relative_probe_error"] for x in rows],
                    "o-",
                    label=f"Vitesse, n={n}",
                )
                axes[0].semilogy(
                    [x["band"] for x in rows],
                    [max(x["relative_l2"]) for x in rows],
                    "s--",
                    label=f"Champ, n={n}",
                )
            for steps, style in ((8, "--"), (32, "-")):
                rows = sorted(
                    [
                        x
                        for x in broad
                        if x["nfe"] == n
                        and x["method"] == "memory"
                        and x["steps"] == steps
                    ],
                    key=lambda x: x["band"],
                )
                axes[1].loglog(
                    [x["evolution_seconds"] for x in rows],
                    [max(x["relative_l2"]) for x in rows],
                    "o" + style,
                    label=f"n={n}, {steps} pas",
                )
        axes[0].set(
            xlabel="Rayon spectral K",
            ylabel="Erreur relative",
            title="Vitesse filtrée et champ final",
        )
        axes[1].set(
            xlabel="Évolution GPU (s, hors préparation)",
            ylabel="Erreur relative L2 du champ",
            title="Prix du raffinement spectral",
        )
        for ax in axes:
            ax.grid(True, which="both", alpha=0.18)
            ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-hamiltonian-broad",
            "Fonction de courant non limitée en fréquence. Composition de Yoshida à sept stages. Le plateau éventuel combine erreur spatiale, temporelle et projections de cartes. Les temps de cette étude sont des mesures exploratoires uniques hors graphes CUDA, non comparables directement aux médianes du comparatif précédent.",
        )
        r.table(
            ["n FE", "K", "Modes", "Pas", "Erreur vitesse", "Erreur champ"],
            [
                [
                    x["nfe"],
                    x["band"],
                    x["modes"],
                    x["steps"],
                    x["velocity_relative_probe_error"],
                    max(x["relative_l2"]),
                ]
                for x in broad
                if x["method"] == "memory" and x["steps"] == 32
            ],
        )
        r.paragraph(
            f"Le raffinement des caractéristiques change les pieds d’au plus {max(x['reference_refinement_error'] for x in broad):.3g}; l’invariance de la fonction de courant le long de ces trajectoires est satisfaite à {max(x['reference_invariant_error'] for x in broad):.3g} près. Le défaut relatif maximal des moments FE contre la FFT analytique vaut {max(x['fourier_moment_relative_error'] for x in broad):.3g}. Il reste nécessaire de contrôler le coût et l’erreur pour des vitesses FE non lisses, dépendant du temps ou issues d’un solveur couplé."
        )
    spatial = [
        x["result"]
        for degree in (1, 2, 4)
        for x in raw.get(f"hamiltonian-spatial-p{degree}", [])
    ]
    if spatial:
        r.section("Raffinement spatial aux degrés un, deux et quatre")
        r.paragraph(
            "Cette campagne conserve huit pas de transport composé et raffine simultanément le maillage FE et le support auxiliaire, dont la résolution est deux fois plus grande. Les neuf moments de vitesse sont recalculés depuis la fonction de courant FE à chaque degré. Les erreurs sont mesurées contre les mêmes caractéristiques analytiques; le plancher de projection FE est évalué indépendamment."
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        spatial_orders = []
        for ax, degree in zip(axes, (1, 2, 4)):
            for method, label in (
                ("memory", "Mémoire"),
                ("reset", "Reconstruction FE"),
                ("graph", "Graphe orthogonal"),
            ):
                selected = sorted(
                    [
                        x
                        for x in spatial
                        if x["degree"] == degree and x["method"] == method
                    ],
                    key=lambda x: x["nfe"],
                )
                if selected:
                    ax.loglog(
                        [x["nfe"] for x in selected],
                        [max(x["relative_l2"]) for x in selected],
                        "o-",
                        label=label,
                    )
                    if len(selected) > 1:
                        a, b = selected[-2:]
                        spatial_orders.append(
                            [
                                degree,
                                label,
                                a["nfe"],
                                b["nfe"],
                                math.log(
                                    max(a["relative_l2"]) / max(b["relative_l2"]),
                                    b["nfe"] / a["nfe"],
                                ),
                            ]
                        )
            selected = sorted(
                [
                    x
                    for x in spatial
                    if x["degree"] == degree and x["method"] == "memory"
                ],
                key=lambda x: x["nfe"],
            )
            if selected:
                ax.loglog(
                    [x["nfe"] for x in selected],
                    [max(x["relative_fe_floor"]) for x in selected],
                    ":",
                    color="#555555",
                    label="Projection FE",
                )
            ax.set(
                xlabel="Résolution FE n",
                ylabel="Erreur relative L2 maximale",
                title=f"Degré {degree}",
            )
            ax.grid(True, which="both", alpha=0.18)
        axes[0].legend(fontsize=7)
        r.save_figure(
            fig,
            "night-hamiltonian-spatial",
            "Neuf directions de vitesse, huit pas de Yoshida, supports uniformes de résolution 2n. Les pentes mesurées sont propres à ce cas lisse; aucune convergence générale pour toute vitesse FE n’est inférée.",
        )
        r.table(
            ["Degré", "Méthode", "n initial", "n final", "Ordre observé"],
            spatial_orders,
        )
    reaction = [
        x["result"]
        for name in (
            "hamiltonian-reaction",
            "hamiltonian-reaction-fine",
            "hamiltonian-reaction-target",
        )
        for x in raw.get(name, [])
    ]
    if reaction:
        r.section("Réaction physique entre les nouveaux transports")
        r.paragraph(
            "La même vitesse à neuf directions transporte maintenant deux champs soumis au taux spatial κ(1 + 0,5 sin(2πx)), jusqu’à T = 0,25. Les demi-pas de réaction sont positifs et encadrent le transport composé de Yoshida. L’ordre global attendu de cette séparation réaction/transport est deux : les sous-pas négatifs concernent uniquement l’advection. La référence RK4 indépendante intègre simultanément les pieds et l’exposant de réaction; le cas de vitesse nulle vérifie séparément le signe et la valeur du taux."
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        labels = {
            "full": "Moments complets",
            "target": "Moment cible retenu",
            "reset": "Reconstruction FE",
            "fe-only": "Réaction FE seule",
        }
        for row, n in enumerate((64, 128)):
            for col, strength in enumerate((1, 100)):
                ax = axes[row, col]
                for method, label in labels.items():
                    selected = sorted(
                        [
                            x
                            for x in reaction
                            if x["nfe"] == n
                            and x["strength"] == strength
                            and x["method"] == method
                        ],
                        key=lambda x: x["steps"],
                    )
                    if selected:
                        ax.loglog(
                            [x["steps"] for x in selected],
                            [max(x["relative_l2"]) for x in selected],
                            "o-",
                            label=label,
                        )
                selected = [
                    x for x in reaction if x["nfe"] == n and x["strength"] == strength
                ]
                if selected:
                    ax.axhline(
                        max(selected[0]["relative_fe_floor"]),
                        color="#555555",
                        ls=":",
                        label="Projection FE",
                    )
                ax.set(
                    xlabel="Nombre de pas",
                    ylabel="Erreur relative L2 (max. des 2 champs)",
                    title=f"n FE = {n}, force {strength}",
                )
                ax.grid(True, which="both", alpha=0.18)
        axes[0, 0].legend(fontsize=8)
        r.save_figure(
            fig,
            "night-hamiltonian-reaction",
            "Réaction réellement appliquée entre les transports. Les moments complets sont comparés à une reconstruction FE cohérente et à l’extension incomplète qui laisse la mémoire sans réaction. Ce dernier modèle peut avoir une erreur relative élevée malgré la petite amplitude absolue finale.",
        )
        r.table(
            ["n FE", "Force", "Méthode", "Pas", "Erreur max.", "Évolution (s)"],
            [
                [
                    x["nfe"],
                    x["strength"],
                    labels[x["method"]],
                    x["steps"],
                    max(x["relative_l2"]),
                    x["evolution_seconds"],
                ]
                for x in reaction
                if x["steps"] == 512
            ],
        )
        r.paragraph(
            f"Le plus grand changement relatif de la référence sous raffinement vaut {max(x['reference_refinement_relative_error'] for x in reaction):.3g}. Les énergies finales sont évaluées par quadrature positive des intersections. Le coût ci-dessus est une mesure unique d’évolution hors graphes CUDA, après construction des opérateurs et hors diagnostics finaux; les préparations du transport et de la réaction sont enregistrées séparément. Une amélioration de précision à maillage fixé ne démontre pas un gain à coût égal, ni l’utilité indispensable de la mémoire."
        )
        field = r.output / "night-hamiltonian-fields-reaction100-yoshida6-128-512.pdf"
        if field.exists():
            r.figure(
                field.stem,
                "Premier champ sous réaction forte : référence et erreurs absolues, avec une échelle logarithmique commune aux méthodes. Les erreurs relatives de chaque composante sont données séparément dans les données.",
            )
    evidence = [
        r"\paragraph{Filtered FE velocity and signed shear compositions.}",
        r"Nine Fourier orientations are obtained from a projected $P_4$ streamfunction on perturbed triangles. Two smooth periodic profiles are transported to $T=0.5$. Independent RK4 characteristics refine to a foot discrepancy below $5.2\,10^{-14}$. The same FE space and filtered velocity drive a central/upwind DG--RK4 comparator. All evolution operators use CUDA graphs, including the FE observation at every semi-Lagrangian step.",
    ]
    if orders:
        evidence.append(
            "The measured orders between two and four time steps at $n=128$ are "
            + ", ".join(f"{x[3]:.2f}" for x in orders)
            + r" for the three-stage fourth-order, five-stage fourth-order and seven-stage sixth-order compositions. These are pre-plateau measurements, not uniform temporal orders at fixed discrete space."
        )
    if matched_summary:
        evidence += [
            r"\begin{center}\small\begin{tabular}{lrrr}\toprule Method & Steps & Relative error & Evolution (s)\\\midrule"
        ]
        for x in matched_summary:
            label = {
                "dg-central": "DG central",
                "dg-upwind": "DG upwind",
                "memory": "Memory/Y6",
                "reset": "Reset/Y6",
            }[x["method"]]
            evidence.append(
                f"{label} & {x['steps']} & {x['relative_error']:.3e} & {x['median_seconds']:.4f}"
                + r"\\"
            )
        evidence += [
            r"\bottomrule\end{tabular}\end{center}",
            r"The table reports medians of five blocks with alternating case order, excluding assembly, graph capture and the independent reference. The report provides the full timing ranges. Similar performance of reset and persistent memory would attribute the speed advantage to the transport composition, not to persistent memory. No intermediate reaction is included in this comparison.",
        ]
    if benchmarks:
        evidence += [
            r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{night-hamiltonian-cost.pdf}",
            r"\caption{Accuracy versus evolution cost for identical FE spaces and velocities. Solid lines preserve memory; dashed lines reconstruct it from the FE observation after every step. Preparation costs are excluded.}\end{figure}",
        ]
    if broad:
        selected = [
            x
            for x in broad
            if x["nfe"] == 128
            and x["band"] == 10
            and x["steps"] == 8
            and x["method"] == "memory"
        ]
        evidence.append(
            r"A second streamfunction, $0.04\exp(0.7\cos(2\pi(x-0.5))+0.45\cos(2\pi(y-0.5))-1.15)$, has infinitely many Fourier modes. Its FE moments are truncated to a square spectral band and checked against independent analytic-function FFT moments. The reference integrates the original, untruncated analytic velocity."
        )
        if selected:
            x = selected[0]
            evidence.append(
                f"At spectral radius ten, $n=128$ and eight steps, the maximum component relative error is {max(x['relative_l2']):.3e}, against an FE projection floor {max(x['relative_fe_floor']):.3e}; the sampled relative velocity error is {x['velocity_relative_probe_error']:.3e}. This uses {x['modes']} retained wavevectors. Increasing the spectral bandwidth substantially increases cost, and this study alone establishes no efficiency advantage for general velocities."
            )
        evidence += [
            r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{night-hamiltonian-broad.pdf}",
            r"\caption{Spectral truncation of a non-bandlimited FE streamfunction. Analytic characteristics separate velocity filtering from transport error.}\end{figure}",
        ]
    (r.output / "hamiltonian_evidence.tex").write_text("\n".join(evidence) + "\n")
