"""Build the follow-up scientific report from the incremental GPU campaign.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
from datetime import datetime
import html
import json
import math
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from report import Report, load, successful, tex_escape, apply_style, plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_style()
    suites = ("validation", "convergence", "small_steps", "long_time", "resolution")
    raw = {s: load(args.input, "carrier-"+s) for s in suites}
    data = {s: successful(v) for s, v in raw.items()}
    old = successful(load(args.baseline, "deformation"))
    r = Report(args.output)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y à %H:%M")
    r.paragraph("Complément au rapport de nuit. Généré le "+stamp+", heure de Paris. "
                "Les résultats ci-dessous concernent un nouvel algorithme, implémenté après la campagne initiale.")
    r.section("Changement algorithmique")
    r.paragraph("Le champ reste représenté dans les éléments finis triangulaires. Un support cartésien DG "
                "auxiliaire reçoit une projection L2 du champ, effectue le transport, puis retourne seulement "
                "l'incrément. Les échanges FE–support sont des projections adjointes préassemblées par "
                "intersection statique. Aucun ajustement global de moments ni intersection de cellules "
                "transportées n'intervient pendant les pas de cette variante.")
    r.body.append("<p><strong>T = I + R* (S − I) R</strong>, avec R la projection FE vers le support et R* son adjoint de masse.</p>")
    r.tex.append(r"\[\boxed{T=I+R^*(S-I)R.}\]")
    r.paragraph("Si R est contractant et préserve la constante, et si S conserve la masse, la constante "
                "et la norme L2 au sens large, le transfert T conserve la masse et la constante et est "
                "contractant. Il vaut exactement l'identité lorsque S vaut l'identité. La note "
                "incremental_theory.pdf donne une preuve et la décomposition de la dissipation. "
                "Le transfert est linéaire et ne comporte pas de limiteur dépendant du champ.")
    r.paragraph("Ces hypothèses sont réalisées par le prototype pour la translation et le cisaillement "
                "horizontal incompressible, en double précision sur RTX 5090. Les projections statiques "
                "couplent des triangles affines Pp à des carrés Qp ; le transport auxiliaire est une "
                "projection DG 1D par ligne de noeuds de Gauss, puis par colonne pour la translation 2D. "
                "La vitesse peut varier non linéairement entre les lignes dans le cas du cisaillement.")

    r.section("Contrôles indépendants")
    failed = sum(x["status"] != "ok" for rows in raw.values() for x in rows)
    r.paragraph(f"{sum(map(len, raw.values()))} cas enregistrés, {failed} échecs d'exécution. "
                "Les six validations combinent P1, P2, P3 et deux résolutions auxiliaires. "
                "Elles comprennent une seconde quadrature des intersections, les moments exacts de Gauss, "
                "l'adjoint comparé à une matrice dense indépendante, les constantes, la masse, "
                "la contraction de R et 48 SVD des transferts complets, dont les identités à pas nul.")
    r.table(["Degré", "H = h / facteur", "Norme R", "Noyau minimal", "Norme T, max."],
            [[x["p"], x["factor"], x["projection_singular_max"], x["projection_nullity"],
              max(z["singular_max"] for z in x["spectra"])] for x in data["validation"]])
    sanitizer = args.input/"carrier-sanitizer.log"
    if sanitizer.exists():
        lines = [x for x in sanitizer.read_text().splitlines() if "ERROR SUMMARY" in x]
        r.paragraph("Compute Sanitizer : "+(" ; ".join(lines) if lines else "aucun résultat final disponible"))

    r.section("Précision sur un flot non affine")
    selected = sorted([x for x in data["convergence"] if x["p"] == 3 and x["flow"] == "shear"], key=lambda x: x["n"])
    baseline = sorted([x for x in old if x["p"] == 3 and x["flow"] == "shear" and x["steps"] == 16], key=lambda x:x["n"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog([1/x["n"] for x in baseline], [x["l2"][0] for x in baseline], "s-", label="Sommets transportés, projection géométrique")
    ax.loglog([1/x["n"] for x in selected], [x["l2"][0] for x in selected], "o-", label="Transfert incrémental, support DG")
    ax.set(xlabel=r"$h=1/n$", ylabel=r"$\|q_h-q\|_{L^2}$", title="Cisaillement, P3, 16 pas, durée finale 1")
    ax.grid(True, which="both", alpha=.2); ax.legend(fontsize=9)
    r.save_figure(fig, "incremental-convergence", "Même champ, même maillage triangulaire et même durée. La référence géométrique utilise les images affines des cellules ; le nouveau support évalue le cisaillement aux noeuds de Gauss.")
    by_n = {x["n"]:x for x in baseline}
    r.table(["n", "Erreur géométrique", "Erreur incrémentale", "Rapport erreurs", "Défaut masse"],
            [[x["n"], by_n[x["n"]]["l2"][0], x["l2"][0], by_n[x["n"]]["l2"][0]/x["l2"][0], x["mass_error"]] for x in selected])
    if len(selected)>1:
        rate=math.log(selected[-2]["l2"][0]/selected[-1]["l2"][0],2)
        r.paragraph(f"Le dernier taux de raffinement P3 est {rate:.3f}. Les erreurs sont intégrées "
                    "avec la quadrature indépendante de degré 18. Cette amélioration corrige principalement "
                    "la limitation géométrique de la référence sur ce flot ; elle ne mesure pas un gain face "
                    "à toutes les méthodes SLDG utilisant une géométrie courbe d'ordre élevé.")

    r.section("Petits pas et temps long")
    fig, ax = plt.subplots(figsize=(8, 4))
    for flag, name in [(True, "Retour de l'incrément"), (False, "Retour du champ entier")]:
        rows = sorted([x for x in data["small_steps"] if x["p"] == 3 and x["incremental"] == flag],key=lambda x:x["steps"])
        ax.loglog([x["steps"] for x in rows], [x["l2"][0] for x in rows], "o-", label=name)
    ax.set(xlabel="Nombre de pas, durée finale fixée à 0,1", ylabel="Erreur L2", title="P3, n=16 : effet du retour de l'incrément")
    ax.grid(True, which="both", alpha=.2); ax.legend()
    r.save_figure(fig, "incremental-small-steps", "Les deux variantes emploient exactement les mêmes projections statiques et le même transport auxiliaire. Le retour du champ entier applique une double projection à chaque pas.")
    r.table(["Degré", "Facteur", "Pas", "Erreur L2", "Défaut masse", "Énergie finale / initiale"],
            [[x["p"],x["factor"],x["steps"],x["l2"][0],x["mass_error"],x["energy_ratio"]] for x in data["long_time"]])
    r.paragraph("Le temps long correspond à une translation pendant 10 unités de temps, avec 10 000 pas "
                "et n=32. Les différences à l'arrondi ne sont pas corrigées par un rétablissement global "
                "de masse ou une renormalisation d'énergie.")

    r.section("Résolution auxiliaire et coût")
    r.paragraph("À résolution égale et degré égal, l'espace cartésien possède moins de degrés de liberté "
                "que l'espace triangulaire. Certains modes FE appartiennent nécessairement au noyau de R "
                "et restent immobiles sous T. Les SVD avec un support deux fois plus fin dans chaque "
                "direction ne détectent plus ce noyau sur les petits maillages testés. Le choix du support "
                "fait donc partie de la méthode ; le théorème d'énergie ne suffit pas à le choisir.")
    rows=[x for x in data["resolution"] if x["n"] == 64]
    r.table(["Facteur", "Erreur L2", "Couplage (Mo)", "Préassemblage (s)", "Application (ms)"],
            [[x["factor"],x["l2"][0],x["coupling_bytes"]/1e6,x["static_setup_seconds"],x["application_ms"]] for x in rows])
    r.paragraph("Les applications sont chronométrées par événements CUDA après échauffement, avec "
                "allocations du prototype et lancements Python. Le préassemblage statique est mesuré "
                "séparément et peut inclure des coûts de cache. Aucune accélération globale face à un "
                "solveur eulérien GPU n'est établie. Pour une vitesse variable, les poids du transport "
                "auxiliaire changent, mais les couplages entre maillages restent réutilisables.")

    r.section("Ce qui reste à établir pour un papier")
    r.paragraph("La construction donne maintenant un algorithme linéaire exécuté sur GPU, une preuve "
                "conditionnelle de conservation et de contraction, l'identité à déplacement nul et des "
                "résultats d'ordre élevé sur un flot non affine. Il reste à démontrer une borne de "
                "consistance uniforme, traiter les modes mal résolus par le support et construire un "
                "transport auxiliaire adapté aux vitesses FE générales. Les frontières, la compressibilité "
                "et le calcul distribué ne sont pas implémentés dans cette variante.")
    r.paragraph("Le retour d'incréments a des antécédents dans FLIP ; les projections de masse "
                "conservatives et stables sont également établies. Une nouveauté ne peut pas être "
                "revendiquée sur ces seuls ingrédients. Le positionnement doit porter sur leur réalisation "
                "FE statique, les propriétés de l'opérateur complet, le contrôle du support et le coût à "
                "erreur fixée. La preuve et les références sont jointes dans incremental_theory.pdf.")
    r.body.append("<p><a href='https://arxiv.org/html/1610.05258v1'>Duque et Español, Assignment of fields from particles to mesh</a> ; <a href='https://arxiv.org/abs/2406.01936'>Fluid Implicit Particles on Coadjoint Orbits</a>.</p>")
    r.tex.append(r"\url{https://arxiv.org/html/1610.05258v1}; \url{https://arxiv.org/abs/2406.01936}.")
    r.section("Autres pistes examinées")
    r.paragraph("Les bandes de quadrature adaptées aux faces rendent faisables les premiers cas de "
                "petit déplacement qui échouaient avec des points intérieurs fixes, mais le solveur "
                "complet de moments reste coûteux. La correction locale des Gram après équilibrage "
                "des seules masses assure des invariants mais a échoué au test de convergence : elle "
                "est conservée comme expérience rejetée. Une variante par flux caractéristiques aux "
                "faces donne aussi une bonne précision en cisaillement ; son contrôle d'énergie "
                "est non linéaire. Elle constitue une comparaison exploratoire distincte du transfert "
                "linéaire incrémental présenté ici.")
    r.paragraph("Reproduction : python3 experiments/slfe/carrier_study.py --suite validation --output DIR, "
                "puis les suites convergence, small_steps, long_time et resolution. Les JSONL contiennent "
                "les paramètres, les versions et les empreintes SHA-256 des sources.")
    title="SLFE : transport incrémental par espace auxiliaire"
    style="body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#152c3a}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #cbd5de;padding:8px;text-align:left}img{max-width:100%}figcaption{font-size:14px}"
    (args.output/"rapport_increments.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'><title>"+title+"</title><style>"+style+"</style><h1>"+title+"</h1>"+"\n".join(r.body))
    preamble=r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\usepackage[french]{babel}
\usepackage{amsmath,amssymb,geometry,graphicx,booktabs,hyperref}
\geometry{margin=2cm}\hypersetup{colorlinks=true,urlcolor=blue,hypertexnames=false}
\setlength{\emergencystretch}{3em}
\title{SLFE : transport incrémental\\par espace auxiliaire}
\author{Prototype GPU et résultats complémentaires}\date{DATE}
\begin{document}\maketitle
""".replace("DATE",tex_escape(stamp))
    (args.output/"rapport_increments.tex").write_text(preamble+"\n".join(r.tex)+r"\end{document}")
    with (args.output/"pdf-build.log").open("w") as log:
        for _ in range(2):
            subprocess.run(["lualatex","-interaction=nonstopmode","-halt-on-error","rapport_increments.tex"],cwd=args.output,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=dict(generated=stamp,counts={s:len(v) for s,v in raw.items()},execution_failures=failed,
                 max_mass_error=max(x["mass_error"] for s in suites if s != "validation" for x in data[s]))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for path in args.output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
