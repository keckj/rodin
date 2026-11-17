"""Produce a scientific PDF and an HTML report from durable GPU measurements.

This is host-side analysis and presentation, not a numerical CPU solver.
Distributed under the Boost Software License, Version 1.0.
"""
import argparse
from collections import Counter
from datetime import datetime
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from plot_style import apply_style


def load(directory, name):
    file = directory/(name+".jsonl")
    if not file.exists():
        return []
    records = []
    for line in file.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrently written trailing line is picked up next time.
            continue
    return records


def successful(rows):
    return [r["result"] for r in rows if r.get("status") == "ok"]


def num(value):
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        return num(value[0]) if value else "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return str(value)
    return f"{value:.4g}"


def tex_escape(text):
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(table.get(c, c) for c in str(text))


class Report:
    def __init__(self, output):
        self.output = output
        self.body = []
        self.tex = []

    def section(self, title):
        self.body.append(f"<h2>{html.escape(title)}</h2>")
        self.tex.append(r"\section{"+tex_escape(title)+"}")

    def paragraph(self, text):
        self.body.append(f"<p>{html.escape(text)}</p>")
        self.tex.append(tex_escape(text)+"\n")

    def table(self, headers, rows):
        self.body.append("<div class='table'><table><thead><tr>"+"".join("<th>"+html.escape(x)+"</th>" for x in headers)+"</tr></thead><tbody>")
        self.tex.append(r"\begin{center}\small\begin{tabular}{"+"l"*len(headers)+r"}\toprule")
        self.tex.append(" & ".join(tex_escape(x) for x in headers)+r"\\\midrule")
        for row in rows:
            values = [num(x) for x in row]
            self.body.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in values)+"</tr>")
            self.tex.append(" & ".join(tex_escape(x) for x in values)+r"\\")
        self.body.append("</tbody></table></div>")
        self.tex.append(r"\bottomrule\end{tabular}\end{center}")

    def figure(self, name, caption):
        self.body.append(f"<figure><img src='{name}.svg' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>")
        self.tex.append(r"\begin{figure}[htbp]\centering\includegraphics[width=.95\linewidth]{"+name+r".pdf}\caption{"+tex_escape(caption)+r"}\end{figure}")

    def save_figure(self, fig, name, caption):
        fig.tight_layout()
        fig.savefig(self.output/(name+".svg"), transparent=True)
        fig.savefig(self.output/(name+".pdf"), transparent=True)
        plt.close(fig)
        self.figure(name, caption)


def generate(source, output, final=False):
    output.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"figure.figsize": (8.5, 4.5), "font.size": 10})
    suites = ["translation", "spectrum", "cartesian", "performance", "long_time",
              "deformation", "moments", "curved_moments", "robustness", "support"]
    raw = {name: load(source, name) for name in suites}
    data = {name: successful(rows) for name, rows in raw.items()}
    r = Report(output)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y à %H:%M:%S")
    status = "Rapport de campagne" if final else "Rapport provisoire, campagne en cours"
    r.paragraph(status+". Généré le "+stamp+" (Europe/Paris).")
    counts = {name: len(rows) for name, rows in raw.items()}
    failures = sum(row.get("status") != "ok" for rows in raw.values() for row in rows)
    r.paragraph(f"{sum(counts.values())} expériences enregistrées sur RTX 5090, dont {failures} échecs d'exécution. "
                "Un calcul terminé n'est pas nécessairement une méthode validée : les résidus et les poids sont contrôlés séparément.")

    r.section("Ce que les expériences permettent de conclure")
    r.paragraph("La voie semi-lagrangienne FE fonctionne sur les triangles testés : le transfert géométrique GPU "
                "conserve la masse et retrouve les ordres spatiaux attendus pour la translation. Le dépôt direct de "
                "points de quadrature peut conserver la masse tout en amplifiant des perturbations et en perdant la précision.")
    r.paragraph("Une construction par cubature commune à moments compatibles a été implémentée. Sous deux identités "
                "de Gram et des poids non négatifs, une preuve donne conservation, reproduction de la constante et "
                "contraction en norme L2 pour un flot incompressible. Les petits cas acceptés illustrent ce mécanisme. "
                "Le coût et le conditionnement de la construction restent les obstacles principaux.")
    r.paragraph("Une obstruction mathématique précise a aussi été identifiée : à déplacement non nul, "
                "une cellule qui n'échange aucune particule ne peut satisfaire simultanément les moments "
                "de degré un au départ et à l'arrivée, même avec des poids signés. Un support fixe intérieur "
                "aux éléments est donc insuffisant lorsque le pas tend vers zéro. Il faudra adapter les points aux interfaces et au flot.")
    r.paragraph("Aucune nouveauté révolutionnaire n'est établie par cette campagne. Le résultat utile pour la suite "
                "est un choix de problème précis : construire cette compatibilité sans intersection de cellules "
                "courbes, avec un coût local ou quasi linéaire et un contrôle explicite de la faisabilité. "
                "Les travaux sur les remaps conservatifs, les projections compatibles et l'optimisation imposent une comparaison approfondie.")

    r.section("Périmètre réel et reproductibilité")
    r.paragraph("Toutes les simulations, localisations, intersections, projections, SVD et réductions numériques "
                "sont exécutées sur CUDA via CuPy. Symox génère et vérifie les bases et noyaux avant les calculs. "
                "Python pilote les lancements ; les figures et ce rapport sont produits sur l'hôte. Le code est un "
                "prototype de recherche dans experiments/slfe, pas encore un opérateur intégré au langage variationnel de Rodin.")
    r.paragraph("Le domaine est un tore périodique 2D. Les maillages contiennent 2 n² triangles affines, avec "
                "sommets perturbés et diagonales variables. Le localisateur exploite leur origine en grille : il "
                "n'est pas un BVH pour maillages importés quelconques. Les espaces DG P0 à P3 sont modaux, "
                "orthonormaux en moyenne sur le triangle de référence. Le calcul est en double précision, sur un GPU et sans MPI.")
    r.paragraph("Les profils lisses, trajectoires et références sont définis dans gpu.py et flows.py. "
                "Les erreurs FE sont intégrées par une quadrature de degré 18 indépendante de celle du transfert. "
                "Les extrema sont échantillonnés, et ne sont pas des bornes mathématiques certifiées. "
                "Les fichiers JSONL conservent paramètres, durées, échecs et empreintes SHA-256 du code.")
    r.table(["Campagne", "Calculs", "Échecs"], [[k, len(v), sum(x.get("status") != "ok" for x in v)] for k, v in raw.items() if v])

    r.section("Validation indépendante")
    for name in ["validation", "validation-moments", "geometry-control"]:
        file = source/(name+".json")
        if file.exists():
            result = json.loads(file.read_text())
            checks = result.get("checks", result.get("records", []))
            r.paragraph(f"{name} : statut {result.get('passed')}, {len(checks)} contrôles enregistrés.")
        else:
            r.paragraph(f"{name} : aucun certificat de fin disponible à la génération de ce rapport.")
    sanitizer = source/"sanitizer.log"
    if sanitizer.exists():
        text = sanitizer.read_text()
        lines = [x for x in text.splitlines() if "ERROR SUMMARY" in x]
        r.paragraph("Compute Sanitizer : "+(" ; ".join(lines[-2:]) if lines else "journal présent, résultat à inspecter"))
    r.paragraph("Les contrôles comprennent les moments de quadrature contre une formule factorielle exacte, "
                "l'orthogonalité des bases, l'identité sans déplacement, les constantes, les bilans de masse, "
                "l'identité de l'adjoint, une seconde quadrature des intersections et le raffinement spatial. "
                "Une perte artificielle de 1 % de masse est rejetée par les tests. Les hypothèses de positivité "
                "et de compatibilité du candidat sont vérifiées avant toute interprétation de sa contraction.")

    r.section("Translation : ordre, masse et temps long")
    fig, ax = plt.subplots()
    rows = []
    for p in [1, 2, 3]:
        selected = sorted([x for x in data["translation"] if x["mode"] == "geometric" and x["p"] == p and x["steps"] == 64], key=lambda x: x["n"])
        if selected:
            ax.loglog([1/x["n"] for x in selected], [x["l2"][0] for x in selected], "o-", label=f"DG P{p}")
            last = selected[-1]
            rate = math.log(selected[-2]["l2"][0]/last["l2"][0], 2) if len(selected)>1 else None
            rows.append([p, last["n"], last["l2"][0], rate, last["mass_absolute_error"][0]])
    ax.set(xlabel=r"$h=1/n$", ylabel=r"$\|q_h-q\|_{L^2}$", title="Translation, 64 transferts")
    ax.grid(True, which="both", alpha=.2); ax.legend()
    r.save_figure(fig, "translation", "Raffinement du maillage, même durée finale et même nombre de transferts. Les trajectoires de translation sont exactes.")
    r.table(["Degré", "n", "Erreur L2", "Taux final", "Défaut masse"], rows)
    long = [x for x in data["long_time"] if x["n"] == 128 and x["steps"] == 10000]
    if long:
        r.table(["Degré", "Transferts", "Erreur L2", "Défaut masse", "Énergie finale / initiale"],
                [[x["p"], x["steps"], x["l2"], x["mass_absolute_error"], x["energy_ratio"]] for x in long])
        r.paragraph("Le temps long utilise une bosse périodique lisse et une durée finale de 10. "
                    "Augmenter le nombre de projections peut augmenter l'erreur, même lorsque la trajectoire est exacte.")

    r.section("Le dépôt brut ne suffit pas")
    selected = [x for x in data["translation"] if x["n"] == 64 and x["p"] == 2 and x["steps"] == 128 and x["mode"] == "quadrature"]
    selected.sort(key=lambda x: x["quadrature_per_cell"])
    if selected:
        r.table(["Points / cellule", "Erreur L2", "Défaut masse", "Défaut constante"],
                [[x["quadrature_per_cell"], x["l2"], x["mass_absolute_error"], x["constant_defect"]] for x in selected])
    fig, ax = plt.subplots()
    for p in [0, 1, 2, 3]:
        a = [x for x in data["spectrum"] if x["p"] == p and x["mode"] == "quadrature" and x["qdegree"] == 12]
        a.sort(key=lambda x: x["subdivisions"])
        if a:
            ax.loglog([32*x["subdivisions"]**2 for x in a], [x["singular_max"]-1 for x in a], "o-", label=f"P{p}")
    ax.set(xlabel="Points de dépôt par cellule", ylabel=r"$\sigma_{\max}-1$", title="Amplification en norme de masse, n=8")
    ax.grid(True, which="both", alpha=.2); ax.legend()
    r.save_figure(fig, "amplification", "SVD GPU de l'opérateur normalisé par la matrice de masse. Une valeur singulière supérieure à un établit une amplification possible en un pas, pas à elle seule une divergence asymptotique.")

    r.section("Flots déformants : le rôle du Jacobien")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, flow in zip(axes, ["shear", "compressible"]):
        for p in [1, 2, 3]:
            a = sorted([x for x in data["deformation"] if x["flow"] == flow and x["p"] == p and x["steps"] == 16], key=lambda x: x["n"])
            if a:
                ax.loglog([1/x["n"] for x in a], [x["l2"][0] for x in a], "o-", label=f"P{p}")
        ax.set(xlabel=r"$h$", ylabel="Erreur L2", title="Cisaillement" if flow == "shear" else "Compression")
        ax.grid(True, which="both", alpha=.2); ax.legend()
    r.save_figure(fig, "deformation", "Les images des cellules sont ici approchées par les triangles de leurs sommets transportés. Augmenter le degré du champ ne corrige pas cette approximation géométrique.")
    r.paragraph("Pour un cisaillement exactement incompressible, l'interpolation affine de la carte sur des "
                "triangles perturbés peut avoir un Jacobien différent de un. La densité transportée change alors "
                "même si le champ initial est constant. Le contrôle indépendant utilise la norme exacte avant "
                "projection, somme sur les cellules de aire(K) × (J_K − 1)² / J_K. La projection orthogonale "
                "ne peut qu'abaisser cette norme. Le maillage non perturbé sert de contrôle où ce défaut disparaît pour le cisaillement choisi.")

    r.section("Cubature à moments compatibles : résultat et prix")
    r.paragraph("Les poids sont ajustés pour reproduire les moments jusqu'au degré 2p dans chaque cellule de "
                "départ et d'arrivée. La note theory.pdf contient les hypothèses et la preuve. Les coefficients "
                "du transfert FE peuvent être signés ; la non-négativité des poids de cubature sert à la preuve "
                "d'énergie et ne constitue pas un limiteur de positivité du champ.")
    candidates = data["moments"]
    accepted = [x for x in candidates if x.get("accepted_moments") and x.get("accepted_positive")]
    r.paragraph(f"Sur les {len(candidates)} cas du balayage principal, {len(accepted)} satisfont simultanément "
                "le seuil de résidu relatif de moments (1e-9) et celui de non-négativité des poids (-1e-12 en rapport au poids initial). "
                "Ces nombres incluent les variantes et les échecs ; ils ne définissent pas un taux universel de succès.")
    chosen = [x for x in candidates if x["n"] == 8 and x["p"] == 2 and x["fit"] == "positive" and x["solver"] == "dense"]
    if chosen:
        r.table(["Subdivision", "Résidu moments", "Erreur L2", "L2 géométrique", "Construction (s)"],
                [[x["subdivisions"], x["relative_moment_residual"], x["l2"], x["l2_geometric"], x["build_seconds"]] for x in chosen])
    r.paragraph("La SVD dense établit un diagnostic de faisabilité sur de petits systèmes, mais ne constitue "
                "pas un algorithme compétitif de grand calcul. Le gradient conjugué sur les équations normales "
                "souffre du carré du conditionnement. Une correction grossière par modes de Fourier a été explorée "
                "sur le tore ; elle ne fournit pas à ce stade un préconditionneur général pour domaines non structurés.")
    r.paragraph("La suppression irréversible des sites de poids négatif est une recherche active restreinte. "
                "Son échec ne prouve pas l'infaisabilité du problème positif. Une variante par projections de Dykstra "
                "est conservée comme diagnostic. Les cas dont les moments ne convergent pas ne bénéficient pas de la preuve de stabilité.")
    curved = data["curved_moments"]
    if curved:
        selected = [x for x in curved if x["n"] == 8 and x["p"] == 2 and x["steps"] in (1, 4, 16)]
        table = []
        for x in selected:
            ds = x["moment_diagnostics"]
            residual = max(d["relative_moment_residual"] for d in ds)
            accepted_case = x.get("accepted_moments") and x.get("accepted_positive")
            table.append([x["flow"], x["steps"], x["l2"], residual, "oui" if accepted_case else "non"])
        r.table(["Flot", "Pas", "Erreur L2", "Résidu moments", "Accepté"], table)
        r.paragraph("Les lignes non acceptées documentent un échec de construction. Leur erreur de champ "
                    "n'est pas une mesure de précision d'un opérateur satisfaisant le théorème.")
    support = data["support"]
    if support:
        r.paragraph("L'étude du support sépare le petit déplacement et la réduction du degré des contraintes. "
                    "Une cellule isolée n'échange aucune particule avec ses voisines. Pour une translation "
                    "non nulle, la preuve de theory.pdf établit alors une incompatibilité exacte des premiers moments. "
                    "Un critère numérique à tolérance finie ne remplace pas cette obstruction.")
        table = []
        for p in (1, 2):
            for courant in (.1, .01, .001, .0001):
                group = [x for x in support if x["p"] == p and x["courant"] == courant and x["n"] == 4]
                if group:
                    good = sum(bool(x.get("accepted_moments") and x.get("accepted_positive")) for x in group)
                    table.append([p, courant, f"{good}/{len(group)}", min(x["relative_moment_residual"] for x in group),
                                  min(x.get("isolated_cells", 0) for x in group)])
        r.table(["Degré", "Déplacement / h", "Acceptés", "Meilleur résidu", "Cellules isolées, min."], table)
        relaxed = [x for x in support if x["n"] == 8 and x["moment_degree"] == x["p"]]
        if relaxed:
            r.table(["Degré", "Déplacement / h", "Résidu moments", "Valeur singulière max."],
                    [[x["p"], x["courant"], x["relative_moment_residual"], x.get("singular_max")] for x in relaxed])
            r.paragraph("Imposer seulement les moments de degré p peut préserver masse et constante, "
                        "mais ne garantit pas les identités de Gram de degré 2p ni la contraction.")
    robust = data["robustness"]
    if robust:
        good = [x for x in robust if x.get("accepted_moments") and x.get("accepted_positive")]
        r.paragraph(f"Robustesse : {len(robust)} cas terminés, {len(good)} acceptés selon les deux critères. "
                    "Les graines de maillage, déplacements et densités de particules varient. Les cas restants "
                    "doivent être traités comme des limites de la construction, pas comme des validations.")
        groups = sorted(set((x["p"], x["subdivisions"]) for x in robust))
        r.table(["Degré", "Subdivision", "Cas terminés", "Acceptés"],
                [[p, s, sum(x["p"] == p and x["subdivisions"] == s for x in robust),
                  sum(x["p"] == p and x["subdivisions"] == s for x in good)] for p, s in groups])

    r.section("GPU et référence HySoP")
    r.paragraph("Les noyaux Lambda 2,1 et Lambda 4,2 ont été reconstruits avec Symox à partir des contraintes "
                "mathématiques lues dans HySoP, puis évalués dans des kernels CUDA. Il s'agit d'une référence "
                "cartésienne des mêmes noyaux, pas d'un benchmark du runtime HySoP complet. Les moments exacts "
                "et leur réalisation GPU font l'objet de contrôles distincts.")
    selected = [x for x in data["performance"] if x.get("n") == 256 and x.get("p") in (1, 2, 3)]
    if selected:
        r.table(["Degré", "Champs", "DDL / champ", "Application (ms)", "Blocs (Mo)"],
                [[x["p"], x["fields"], x["dofs_per_field"], x["application_ms"], x["storage_bytes"]/1e6] for x in selected])
    r.paragraph("Les temps d'application sont des médianes d'événements CUDA après échauffement et incluent "
                "la remise à zéro du résultat. Les lancements Python, les caches et la fréquence du GPU peuvent "
                "affecter les petits temps. Les temps de construction uniques peuvent inclure la compilation "
                "paresseuse d'opérations CuPy ; ils ne doivent pas être pris pour des mesures asymptotiques stabilisées.")
    r.paragraph("Pour une vitesse constante, les blocs sont construits une fois puis réutilisés. Un flot "
                "variable peut imposer une reconstruction à chaque pas : il faut alors inclure ce coût, ainsi "
                "que les trajectoires. Aucun gain global face à un solveur eulérien GPU n'est établi ici, et aucune "
                "comparaison CPU/GPU n'a été réalisée.")

    r.section("Positionnement pour un papier ambitieux")
    r.paragraph("La contribution à viser serait une construction de transfert garantissant masse, consistance "
                "et énergie, capable de traiter des caractéristiques courbes sans intersections coûteuses et "
                "avec un coût GPU mesuré à précision comparable. L'égalité des deux Gram fournit un cahier des "
                "charges mathématique ; son établissement efficace est le problème ouvert mis en évidence par cette campagne.")
    r.paragraph("Les prochaines preuves doivent porter sur l'existence de poids pour un support transporté, "
                "la consistance et la convergence, le comportement quand le pas tend vers zéro et le coût du "
                "solveur. Les prochaines expériences doivent dépasser le localisateur de grille, inclure un "
                "vrai concurrent eulérien GPU, des vitesses FE, les frontières et la 3D. Une revendication de "
                "nouveauté doit distinguer cette construction des remaps par optimisation et des projections compatibles existantes.")
    references = json.loads((HERE/"references.json").read_text())
    for reference in references:
        r.body.append(f"<p><a href='{html.escape(reference['url'])}'>{html.escape(reference['title'])}</a>. {html.escape(reference['role'])}</p>")
        r.tex.append(tex_escape(reference["title"])+r". \url{"+reference["url"]+"}. "+tex_escape(reference["role"])+"\n")

    r.section("Artefacts et commandes")
    r.paragraph("Sources : branche SLFE du fork keckj/rodin, répertoire experiments/slfe. "
                "Aucune publication vers cbritopacheco/rodin. Les résultats complets sont dans le répertoire "
                "slfe-results ; les données de synthèse et le rapport sont copiés sur la branche lors de la publication.")
    commands = ["python3 experiments/slfe/symbolic.py", "python3 experiments/slfe/validate.py --output /tmp/slfe-validation.json",
                "python3 experiments/slfe/campaign.py --suite translation --output /tmp/slfe-results",
                "python3 experiments/slfe/research.py --suite moments --output /tmp/slfe-results",
                "python3 experiments/slfe/report.py --input /tmp/slfe-results --output /tmp/slfe-report"]
    r.body.append("<pre>"+html.escape("\n".join(commands))+"</pre>")
    r.tex.append("\\begin{verbatim}\n"+"\n".join(commands)+"\n\\end{verbatim}")
    stylesheet = "body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.65 system-ui;color:#172c3b;background:#fbfcfe}h1{font-size:2.4rem;line-height:1.15}h2{margin-top:2.5em;border-top:1px solid #ccd7df;padding-top:1em}a{color:#096b85}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce3e8;text-align:left}.table{overflow:auto}figure{margin:30px 0}img{max-width:100%}figcaption{font-size:14px;color:#405765}pre{overflow:auto;background:#eaf0f4;padding:20px}"
    title = "SLFE sur RTX 5090 : résultats et pistes de recherche"
    (output/"rapport.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>"+title+"</title><style>"+stylesheet+"</style><body><h1>"+title+"</h1>"+"\n".join(r.body)+"</body></html>")
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb,geometry,graphicx,booktabs,hyperref,float}
\geometry{margin=2cm}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,hypertexnames=false,
pdftitle={SLFE GPU research campaign},pdfauthor={SLFE research experiment}}
\setlength{\emergencystretch}{3em}
\title{SLFE sur RTX 5090\\Résultats et pistes de recherche}
\author{Campagne GPU, branche SLFE}
\date{DATE}
\begin{document}\maketitle\tableofcontents
""".replace("DATE", tex_escape(stamp))
    (output/"rapport.tex").write_text(preamble+"\n".join(r.tex)+"\n\\end{document}\n")
    with (output/"pdf-build.log").open("w") as log:
        for _ in range(2):
            subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", "rapport.tex"],
                           cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = {"generated": stamp, "final": final, "counts": counts, "execution_failures": failures,
               "moment_cases": len(candidates), "accepted_moment_cases": len(accepted),
               "robustness_cases": len(robust), "pdf": str(output/"rapport.pdf")}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    generate(args.input, args.output, args.final)
