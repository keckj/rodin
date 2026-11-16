"""Research checkpoint report for nonseparable transport and FE memory coupling.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from report import Report, apply_style, plt, tex_escape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = (
        "day-validation",
        "day-cellular",
        "day-temporal",
        "day-enrichment",
        "day-nested-validation",
        "day-nested",
        "day-persistent",
        "day-spectral",
        "day-spectral-graph",
        "fourier-validation",
        "fourier-accuracy",
        "fourier-time-order",
        "fourier-autonomous",
        "fourier-compressed",
        "reaction",
        "dg-validation",
        "dg-upwind-operator",
        "dg-accuracy",
        "dg-upwind",
        "dg-cfl",
        "graded",
        "graded-repeated",
        "coupled-benchmark",
        "carrier-build",
        "modes-validation",
        "modes-accuracy",
        "modes-resolution",
        "scale-limits",
        "scale-enrichment",
        "scale-fourier-enrichment",
        "rotation-validation",
        "rotation-accuracy",
        "rotation-refinement",
    )
    raw = {
        name: [
            json.loads(line)
            for line in (args.input / (name + ".jsonl")).read_text().splitlines()
        ]
        for name in names
    }
    data = {
        name: [r["result"] for r in rows if r["status"] == "ok"]
        for name, rows in raw.items()
    }
    memory = json.loads((args.input / "memory-validation.json").read_text())
    failures = sum(r["status"] != "ok" for rows in raw.values() for r in rows)
    apply_style()
    r = Report(args.output)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y à %H:%M")
    r.paragraph(
        "Point d'étape de recherche du "
        + stamp
        + ". La branche SLFE contient des prototypes GPU "
        "distincts de l'interface de production Rodin. La campagne de nuit et le premier "
        "rapport incrémental restent disponibles séparément."
    )
    r.section("Résultat algorithmique")
    r.paragraph(
        "Le transport par projection statique est étendu à un flot cellulaire non séparable. "
        "Trois réalisations sont comparées : des supports DG orientés, un support DG unique "
        "avec changements de coordonnées incrémentaux, puis un support de Fourier unitaire. "
        "Le dernier est couplé aux triangles par deux projections adjointes FE–DG–Fourier."
    )
    r.paragraph(
        "Une différence structurelle apparaît entre le retour FE après chaque pas et la "
        "conservation d'un état auxiliaire. Le retour répété détruit en général l'ordre "
        "temporel élevé à espaces fixés. Conserver la mémoire auxiliaire permet de composer "
        "des sous-transports unitaires d'ordre quatre. Une règle de mise à jour de cette "
        "mémoire permet ensuite d'appliquer une opération physique directement dans l'espace FE."
    )
    r.tex.append(
        r"\[w^+=Sw,\quad v^+=v+R^*(w^+-w),\qquad v^\star=Gv,\quad w^\star=w+R(v^\star-v).\]"
    )
    r.body.append(
        "<p><strong>Advection :</strong> w⁺ = Sw, v⁺ = v + R*(w⁺ − w). "
        "<strong>Physique FE :</strong> v★ = Gv, w★ = w + R(v★ − v).</p>"
    )
    r.paragraph(
        "La note memory_theory.pdf établit une borne d'énergie dans un espace augmenté, "
        "l'ordre temporel du couplage à espaces fixés et des bornes conditionnelles de "
        "consistance. La racine du défaut de projection intervient dans la preuve, "
        "sans être calculée par l'algorithme à deux variables."
    )
    r.section("Champ, domaine et mesures")
    r.paragraph(
        "Le domaine est le tore unité. Le champ initial est une bosse gaussienne périodique. "
        "La fonction de courant est sin²(πx) sin²(πy)/π. Les essais utilisent soit sa vitesse "
        "stationnaire, soit cette vitesse multipliée par cos(πt). Les triangles sont perturbés "
        "et leurs diagonales varient. Les normes FE sont les normes L2 des polynômes reconstruits, "
        "avec une quadrature indépendante pour l'erreur. La solution de référence est obtenue "
        "par caractéristiques RK4 GPU, avec vérification par doublement de la résolution."
    )
    r.paragraph(
        "Les résultats principaux sont évalués à t = 0,5. Le retour à t = 1 du flot réversible "
        "est un contrôle de réversibilité ; il ne suffit pas à mesurer la précision du transport. "
        "Toutes les simulations, projections, décompositions spectrales et réductions numériques "
        "s'exécutent sur RTX 5090 en double précision. Symox prépare les polynômes exacts."
    )
    r.section("Contrôles")
    r.paragraph(
        f"{sum(map(len,raw.values()))} cas principaux sont enregistrés dans les JSONL, "
        f"avec {failures} échec d'exécution. S'y ajoutent quatre validations du couplage "
        "augmenté, six comparaisons des graphes CUDA avec les opérateurs ordinaires et "
        "huit validations FE–Fourier sous Compute Sanitizer."
    )
    sanitizer = (args.input / "sanitizer-fourier.log").read_text()
    r.paragraph(
        "Compute Sanitizer : "
        + " ; ".join(x.strip() for x in sanitizer.splitlines() if "ERROR SUMMARY" in x)
    )
    r.paragraph(
        "Les opérateurs de projection et les transports complets sont vérifiés par SVD "
        "dans la norme de masse sur de petits systèmes. Les adjoints sont comparés à une "
        "transposition dense indépendante. Les moments DG polynomiaux sont comparés à "
        "leur construction par quadrature. Les moments DG–Fourier sont recomputés avec "
        "48 points de Gauss au lieu de 32."
    )
    r.section("Convergence sur le flot cellulaire")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label, steps in (
        ("day-cellular", "DG orientés, 64 pas", 64),
        ("day-nested", "DG imbriqué, 256 pas", 256),
        ("day-persistent", "DG persistant, 256 pas", 256),
    ):
        rows = sorted(
            [
                x
                for x in data[name]
                if x["p"] == 3
                and x["factor"] == 2
                and x["end"] == 0.5
                and x["steps"] == steps
            ],
            key=lambda x: x["n"],
        )
        ax.loglog(
            [1 / x["n"] for x in rows], [x["l2"][0] for x in rows], "o-", label=label
        )
    ax.set(
        xlabel=r"$h=1/n$",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="P3, temps intermédiaire 0,5",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
    r.save_figure(
        fig,
        "cellular-spatial",
        "Le nombre de pas est indiqué pour chaque réalisation. Les courbes combinent erreur spatiale et erreur temporelle.",
    )
    rows = sorted(
        [
            x
            for x in data["fourier-compressed"]
            if x["modes"] == 95 and x["steps"] == 32
        ],
        key=lambda x: (x["p"], x["n"]),
    )
    r.table(
        ["p", "n", "DOF FE", "Erreur L2", "Évolution (s)", "Préparation (s)"],
        [
            [
                x["p"],
                x["n"],
                x["n"] ** 2 * (x["p"] + 1) * (x["p"] + 2),
                x["l2"][0],
                x["evolution_seconds"],
                x["static_setup_seconds"],
            ]
            for x in rows
        ],
    )
    r.paragraph(
        "Ces cas utilisent 95² modes auxiliaires et 32 pas d'ordre quatre avec mémoire. "
        "Le temps d'évolution comprend les échanges FE initiaux et finaux. La préparation "
        "comprend le couplage statique, les plans FFT, l'échauffement et la capture du graphe CUDA. "
        "Les nombres sont des mesures de campagne, sans intervalle statistique."
    )
    r.section("Comparaison à un solveur spectral Eulerien GPU")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparisons = (
        (
            "fourier-compressed",
            "FE–Fourier avec mémoire",
            lambda x: True,
            lambda x: x["l2"][0],
        ),
        (
            "day-spectral-graph",
            "Fourier/RK4, forme antisymétrique",
            lambda x: x["reversing"],
            lambda x: x["l2"],
        ),
    )
    for name, label, select, error in comparisons:
        rows = [x for x in data[name] if select(x)]
        ax.loglog(
            [x["evolution_seconds"] for x in rows],
            [error(x) for x in rows],
            "o",
            alpha=0.65,
            label=label,
        )
    ax.set(
        xlabel="Temps d'évolution (s)",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="Même problème physique, préparation exclue",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
    r.save_figure(
        fig,
        "spectral-cost",
        "Les deux méthodes utilisent les graphes CUDA. Le solveur spectral exploite directement la géométrie périodique ; il ne fournit pas les mêmes inconnues FE.",
    )
    r.paragraph(
        "La référence Fourier/RK4 est compétitive et atteint de très petites erreurs avec "
        "peu d'inconnues sur ce problème lisse. Les essais ne démontrent pas une supériorité "
        "générale du couplage FE. La comparaison suivante utilise un DG direct de même "
        "degré sur les mêmes triangles."
    )
    r.section("DG direct et maillages gradués")
    r.paragraph(
        "Le comparateur DG utilise des flux centrés ou amont et RK4, avec assemblage et "
        "évolution CUDA. Les dérivées des bases sont générées exactement avec Symox. "
        "Les petits opérateurs sont contrôlés par diagonalisation et SVD dans la norme "
        "de masse : conservation, constantes, antisymétrie du flux centré et dissipation "
        "du flux amont. La quadrature est raffinée indépendamment. Compute Sanitizer "
        "ne détecte aucune erreur sur les noyaux DG et le couplage avec réaction."
    )
    r.paragraph(
        "Le pas global DG est augmenté par une campagne de 96 réglages, avec quatre "
        "champs aléatoires et un champ lisse. Un contrôle fini sur ces champs ne constitue "
        "pas un certificat spectral de stabilité. RK4 n'est pas contractant pour tout "
        "générateur dissipatif non normal : les SVD mesurent ici un excès allant jusqu'à "
        "1,61 × 10⁻⁶ pour un pas conservateur. Le premier critère de validation, trop "
        "fort, a été corrigé et son diagnostic est conservé. Ce phénomène est connu "
        "(Sun et Shu, arXiv:1811.10680)."
    )
    r.paragraph(
        "La graduation applique x ↦ x + α sin(2π(x+0,15))/(2π), et une transformation "
        "analogue en y, aux sommets perturbés. Pour α = 0,95 et n = 64, le rapport des "
        "aires extrêmes vaut 2697. Les deux méthodes utilisent 8192 triangles P4, soit "
        "122880 inconnues FE par champ. Le support auxiliaire a 95² valeurs. Le temps "
        "de transport intégré vaut 1/π à t = 0,5 et est partagé uniformément entre les pas."
    )
    repeated = data["graded-repeated"]
    table = []
    for grading in (0.0, 0.95):
        for fields in (1, 4, 16):
            pair = {
                x["method"]: x
                for x in repeated
                if x["grading"] == grading and x["fields"] == fields
            }
            a, d = pair["auxiliary"], pair["dg"]
            table.append(
                [
                    grading,
                    fields,
                    d["l2"][0],
                    a["l2"][0],
                    d["evolution_seconds"],
                    a["evolution_seconds"],
                    d["evolution_seconds"] / a["evolution_seconds"],
                ]
            )
    r.table(
        ["α", "Champs", "L2 DG", "L2 auxiliaire", "DG (s)", "Aux. (s)", "Rapport"],
        table,
    )
    r.paragraph(
        "Temps médians de cinq évolutions, graphes CUDA pour les deux méthodes. "
        "Tous ces cas satisfont une tolérance L2 commune de 10⁻⁶. Les champs multiples "
        "sont des copies du même profil : ce test mesure le débit, pas une diversité "
        "de solutions. Le transport auxiliaire utilise 16 pas. Le dernier réglage DG "
        "testé stable utilise 300 pas sur le maillage non gradué et 3679 sur le maillage "
        "gradué ; le réglage immédiatement supérieur échoue au contrôle aléatoire. "
        "Le gain d'évolution sur le maillage gradué est de 31 à 132 selon le nombre de champs."
    )
    r.paragraph(
        "La préparation est exclue du tableau. Pour le cas gradué à un champ, elle "
        "coûte environ 0,320 s pour l'auxiliaire et 0,014 s pour DG : la première "
        "résolution complète prend donc environ 0,335 s contre 0,496 s, un rapport "
        "de 1,48 seulement. Le gain d'évolution bénéficie surtout à la réutilisation "
        "des opérateurs statiques. Un DG à pas locaux n'a pas été comparé."
    )
    r.paragraph(
        "À 32 pas auxiliaires sur ce maillage gradué, l'erreur vaut 4,130 × 10⁻⁷, "
        "contre 4,126 × 10⁻⁷ pour la meilleure projection FE évaluée par quadrature. "
        "Ce résultat concerne une solution lisse résolue par le support de Fourier. "
        "Il ne garantit pas le transport des petites échelles que seuls les éléments "
        "les plus fins pourraient représenter."
    )
    r.section("Validation de la mémoire et de la physique FE")
    r.table(
        [
            "p",
            "Facteur",
            "Défaut isométrie",
            "Rapport erreur, λ=1",
            "Rapport erreur, λ=10",
        ],
        [
            [
                x["result"]["p"],
                x["result"]["factor"],
                x["result"]["isometry_error"],
                *[z["refinement_ratios"][-1] for z in x["result"]["results"]],
            ]
            for x in memory
        ],
    )
    r.paragraph(
        "Le rapport proche de quatre correspond au dernier doublement du nombre de pas "
        "d'une composition de Strang. La référence est l'exponentielle dense GPU du "
        "générateur augmenté. Les sous-transports sont exacts dans ce contrôle, afin "
        "d'isoler l'erreur de couplage. Un premier essai avec des sous-transports d'ordre "
        "quatre mélangeait les deux erreurs et n'atteignait pas encore ce régime asymptotique ; "
        "le diagnostic est conservé."
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in (16, 32, 64):
        rows = sorted(
            [x for x in data["reaction"] if x.get("memory") is True and x["n"] == n],
            key=lambda x: x["steps"],
        )
        ax.loglog(
            [0.5 / x["steps"] for x in rows],
            [x["l2"][0] for x in rows],
            "o-",
            label=f"P4, n = {n}",
        )
    ax.set(
        xlabel=r"$\Delta t$",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="Advection et réaction FE spatiale",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    r.save_figure(
        fig,
        "reaction-convergence",
        "Le puits vaut λ(x)=1+0,5 sin(2πx). Les exponentielles de réaction sont calculées dans chaque élément. La référence RK4 intègre indépendamment les caractéristiques et leur atténuation.",
    )
    r.paragraph(
        "Le test physique utilise une vitesse stationnaire et une réaction spatialement variable. "
        "La masse décroît physiquement ; la propriété contrôlée est la borne d'énergie, "
        "et l'erreur est évaluée contre la solution caractéristique. Les plafonds d'erreur "
        "aux petits pas diminuent avec le raffinement du maillage."
    )
    r.section("Préparation coopérative et plans CUDA")
    r.paragraph(
        "Une construction coopérative répartit les entrées d'un bloc de projection "
        "entre les 32 threads d'un warp. Vingt-neuf comparaisons couvrent les degrés, "
        "les maillages gradués et plusieurs résolutions auxiliaires. Les blocs P4 "
        "sont identiques bit à bit à l'oracle scalaire dans les cas testés ; les autres "
        "différences restent sous 6e-14. Sur n = 64 gradué, la construction P4 passe "
        "d'une médiane de 297 ms à 98 ms. Les petits blocs conservent le noyau scalaire. "
        "Ces temps concernent la construction des blocs, pas toute la préparation. "
        "Les tableaux précédents conservent leurs mesures antérieures à cette optimisation."
    )
    r.paragraph(
        "Le contrôle mémoire passe. Racecheck ne signale aucun conflit dans le noyau "
        "coopératif ; le contrôle complet signale huit avertissements dans le scan "
        "nonzero de CuPy, conservés dans les diagnostics. Les grands supports ont "
        "révélé séparément une réutilisation défectueuse de plans cuFFT capturés entre "
        "graphes. Chaque graphe possède maintenant ses propres plans. Douze séquences "
        "de changement de taille et quatre sous Compute Sanitizer passent, avec "
        "identité numérique face à l'exécution ordinaire. La campagne concernée a été relancée."
    )
    r.section("Extension aux directions de Fourier entières")
    r.paragraph(
        "Une transformation entière unimodulaire ramène chaque mode incompressible "
        "à un cisaillement, réalisable par permutation et FFT. La somme finie accepte "
        "aussi une vitesse moyenne. La décomposition trigonométrique intégrable a des "
        "antécédents explicites chez Quispel et McLaren (JCP 186, 2003, 308–316). "
        "Les validations contrôlent les constantes, la masse, l'unitarité, la réversibilité, "
        "le générateur dense et une exponentielle GPU indépendante. Les rapports "
        "de raffinement temporel valent 4,000 et 15,996 aux ordres deux et quatre."
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    combined = data["modes-accuracy"] + data["modes-resolution"]
    for grading, label in ((0.0, "Non gradué"), (0.95, "Gradué")):
        rows = sorted(
            [
                x
                for x in combined
                if x["n"] == 64 and x["grading"] == grading and x["steps"] == 64
            ],
            key=lambda x: x["modes"],
        )
        ax.loglog(
            [x["modes"] for x in rows], [x["l2"][0] for x in rows], "o-", label=label
        )
        ax.axhline(rows[-1]["best_projection_error"], linestyle="--", alpha=0.5)
    ax.set(
        xlabel="Points par direction auxiliaire",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="Six modes et vitesse moyenne, P4, n = 64, t = 0,5",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    r.save_figure(
        fig,
        "mode-resolution",
        "Les traits horizontaux indiquent les erreurs des meilleures projections FE. Les paramètres complets des six modes figurent dans les JSONL.",
    )
    r.paragraph(
        "Sur ce flot plus déformant, 95² valeurs auxiliaires ne suffisent plus : "
        "l'erreur plafonne vers 4,56e-3 malgré le raffinement FE. À 767² valeurs, "
        "l'erreur sur le maillage gradué descend à 1,73e-5, proche de la projection "
        "FE à 1,64e-5. L'unitarité n'élimine ni la troncature ni l'aliasing des "
        "changements de coordonnées. Le coût croît avec la résolution et les directions."
    )
    r.section("Contre-exemple : petites échelles résolues seulement en FE")
    r.paragraph(
        "Une bosse de norme L2 proche de un est centrée dans la zone raffinée. "
        "Le transport cellulaire stationnaire dure 0,005. L'erreur est recomputée "
        "avec subdivision de la quadrature. Pour une largeur de 0,0015, la projection "
        "FE finale atteint 4,79e-4 et le DG amont 6,17e-4. Le support auxiliaire "
        "95² ne capture que 22,4 % de l'énergie initiale et donne une erreur de 1,28."
    )
    r.tex.append(
        r"\[\|v(t)-\widehat v(t)\|\geq\max\{0,\|\widehat v(t)-v_0\|-2\|Rv_0\|\}.\]"
    )
    r.body.append("<p>Erreur FE ≥ max(0, déplacement FE requis − 2 ‖Rv₀‖).</p>")
    r.paragraph(
        "La borne démontrée dans la note vaut 0,357 dans ce cas : aucun raffinement "
        "temporel ne peut corriger ce défaut du support fixe. Augmenter seulement "
        "le nombre de modes ne suffit pas non plus lorsque la projection DG "
        "intermédiaire ne résout pas la bosse. Avec un support DG huit fois plus fin "
        "et 1535² valeurs de Fourier, l'erreur descend à 5,74e-4, au prix d'une "
        "augmentation importante du coût. Ce cas est favorable au DG direct."
    )
    r.paragraph(
        "L'avantage mesuré sur les maillages gradués concerne donc les situations "
        "où les petits éléments sont nécessaires à la physique FE ou à la géométrie, "
        "alors que le champ transporté reste résolu sur un support plus économique. "
        "Une méthode générale doit encore traiter efficacement les petites échelles localisées."
    )
    r.section("Piste alternative testée : rotations directement dans le DG")
    r.paragraph(
        "Un second prototype évite le support auxiliaire. Il découpe l'opérateur DG "
        "centré en matrices locales antisymétriques, exponentiées sur GPU. Une rotation "
        "de rang deux corrige chaque sous-opérateur pour préserver la constante. La "
        "composition est orthogonale, conserve la masse et atteint l'ordre quatre. "
        "La note rotation_theory.pdf donne la construction, la preuve et la condition "
        "de regroupement qui évite tout dénominateur singulier, quelle que soit la durée."
    )
    r.paragraph(
        "Six validations denses, vingt-quatre essais d'advection et huit raffinements "
        "confirment les invariants et l'ordre à espace fixé. Cependant, sur n = 32 "
        "gradué, l'erreur reste proche de 0,16 entre 16 et 128 pas. La résolution "
        "temporelle des commutateurs locaux demeure coûteuse. Cette construction "
        "stable n'est donc pas retenue comme solution rapide au problème des petits "
        "éléments. Le résultat négatif est conservé avec son implémentation."
    )
    r.section("Portée scientifique")
    r.table(
        ["α", "Méthode", "Pas", "Erreur L2", "Évolution (s)"],
        [
            [x["grading"], x["method"], x["steps"], x["l2"][0], x["evolution_seconds"]]
            for x in data["coupled-benchmark"]
            if x["n"] == 64
        ],
    )
    r.paragraph(
        "Avec réaction FE, la comparaison emploie la vitesse stationnaire jusqu'à "
        "t = 0,5 et les mêmes matrices locales de réaction. Sur le maillage gradué, "
        "32 pas auxiliaires atteignent 2,10 × 10⁻⁶ en 0,044 s, contre 3,34 × 10⁻⁶ "
        "en 1,016 s pour DG. Sur le maillage non gradué, DG est meilleur à précision "
        "voisine de 4 × 10⁻⁷. Ces temps sont des mesures uniques, préparation exclue ; "
        "le contrôle aléatoire du pas DG a porté sur l'advection seule. Les échecs "
        "initiaux de capture cuBLAS sont archivés ; un noyau CUDA local, comparé "
        "directement au produit matriciel, permet maintenant la capture du couplage."
    )
    r.paragraph(
        "Les ingrédients ont des antécédents : transfert d'incréments FLIP, projections "
        "adjointes, dilatations d'opérateurs, transformations DG–Fourier et compositions "
        "de Yoshida. Aucune priorité scientifique n'est revendiquée à ce stade. La piste "
        "à approfondir est le couplage FE avec mémoire, son analyse de consistance et "
        "sa réalisation rapide, notamment lorsque plusieurs champs partagent le transport."
    )
    r.paragraph(
        "LEoPart applique déjà des incréments de champs FE aux propriétés particulaires "
        "(Maljaars et al., arXiv:1912.13375 ; méthode particles.increment de la documentation). "
        "Le retour de la physique FE vers un état auxiliaire n'est donc pas, à lui seul, "
        "une nouveauté. La question scientifique porte sur les garanties du couplage "
        "contractant, sa consistance et le bénéfice mesuré sur les maillages gradués."
    )
    r.paragraph(
        "Les expériences actuelles couvrent un domaine périodique et des vitesses prescrites "
        "à quatre ou six modes. Les frontières, les vitesses FE arbitraires, la compression, les "
        "flux locaux, le calcul distribué et une preuve générale de convergence restent ouverts. "
        "Les graphes CUDA accélèrent l'exécution sans changer les opérateurs numériques ; "
        "leurs résultats ont été comparés directement à ceux des exécutions ordinaires."
    )
    r.section("Reproduction")
    r.paragraph(
        "Les scripts day_study.py, fourier_study.py, spectral_reference.py, memory_study.py "
        "reaction_study.py, dg_study.py, graded_study.py et coupled_benchmark.py "
        "acceptent un répertoire de sortie explicite. Les JSONL "
        "contiennent les paramètres et les empreintes des sources. Les preuves sont dans "
        "incremental_theory.tex et memory_theory.tex. Aucun code de production Rodin "
        "ni fichier de la bibliothèque HySoP n'est modifié."
    )
    title = "SLFE : flot non séparable et couplage FE avec mémoire"
    style = "body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#152c3a}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #cbd5de;padding:8px;text-align:left}img{max-width:100%}figcaption{font-size:14px}"
    (args.output / "rapport_journee.html").write_text(
        "<!doctype html><html lang='fr'><meta charset='utf-8'><title>"
        + title
        + "</title><style>"
        + style
        + "</style><h1>"
        + title
        + "</h1>"
        + "\n".join(r.body)
    )
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\usepackage[french]{babel}
\usepackage{amsmath,amssymb,geometry,graphicx,booktabs,hyperref}
\geometry{margin=2cm}\hypersetup{colorlinks=true,urlcolor=blue,hypertexnames=false}
\setlength{\emergencystretch}{3em}
\title{SLFE : flot non séparable\\et couplage FE avec mémoire}
\author{Point d'étape de recherche GPU}\date{DATE}
\begin{document}\maketitle
""".replace(
        "DATE", tex_escape(stamp)
    )
    tex = preamble + "\n".join(r.tex) + r"\end{document}"
    for character, replacement in {
        "π": r"\(\pi\)",
        "λ": r"\(\lambda\)",
        "α": r"\(\alpha\)",
        "↦": r"\(\mapsto\)",
        "10⁻⁶": r"\(10^{-6}\)",
        "10⁻⁷": r"\(10^{-7}\)",
    }.items():
        tex = tex.replace(character, replacement)
    (args.output / "rapport_journee.tex").write_text(tex)
    with (args.output / "pdf-build.log").open("w") as log:
        for _ in range(2):
            subprocess.run(
                [
                    "lualatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "rapport_journee.tex",
                ],
                cwd=args.output,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    summary = dict(
        generated=stamp,
        counts={name: len(rows) for name, rows in raw.items()},
        execution_failures=failures,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in args.output.glob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
