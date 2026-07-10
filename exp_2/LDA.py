"""
Analyse LDA des spectres Raman — gélose.

Modalités : dose d'irradiation (0 / 45 Gy) x traitement (NT / +P).
Chaque échantillon (souris) fournit 3 spectres (zones z1/z2/z3) qui restent
groupés lors de la validation croisée : LeaveOneGroupOut est fait par souris
(groups=souris_id), donc les 3 zones d'une même souris ne se retrouvent
jamais séparées entre train et test.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
)

from extract_zone import traiter_acquisitions_gellose, lecteur_données_zones, lecteur_données_moy


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'batch#1': {
        'petri1':  ('S48-G', 45, 'FNT'),
        'petri2':  ('S48-D', 0,  'FNT'),
        'petri3':  ('S38-G', 45, 'FNT'),
        'petri4':  ('S38-D', 0,  'FNT'),
        'petri5':  ('S40-G', 45, 'FNT'),
        'petri6':  ('S40-D', 0,  'FNT'),
        'petri7':  ('S47-G', 45, 'FNT'),
        'petri8':  ('S47-D', 0,  'FNT'),
        #'petri9':  ('S39-G', 0,  'FNT'),
        #'petri10': ('S39-D', 0,  'FNT'),
    },
    'batch#2': {
        'petri11': ('S45-G', 45, 'F+P'),
        'petri12': ('S45-D', 0,  'F+P'),
        'petri13': ('S41-G', 45, 'F+P'),
        'petri14': ('S41-D', 0,  'F+P'),
        'petri15': ('S42-G', 45, 'F+P'),
        'petri16': ('S42-D', 0,  'F+P'),
        'petri17': ('S44-G', 45, 'F+P'),
        'petri18': ('S44-D', 0,  'F+P'),
        'petri19': ('S46-G', 45, 'F+P'),
        'petri20': ('S46-D', 0,  'F+P'),
    },
    # 'batch#3': {
    #     'petri21': ('S33-G', 45, 'MNT'),
    #     'petri22': ('S33-D', 0,  'MNT'),
    #     'petri23': ('S37-G', 45, 'MNT'),
    #     'petri24': ('S37-D', 0,  'MNT'),
    #     'petri25': ('S30-G', 45, 'MNT'),
    #     'petri26': ('S30-D', 0,  'MNT'),
    #     'petri27': ('S32-G', 45, 'M+P'),
    #     'petri28': ('S32-D', 0,  'M+P'),
    #     'petri29': ('S36-G', 45, 'M+P'),
    #     'petri30': ('S36-D', 0,  'M+P'),
    #     'petri31': ('S27-G', 45, 'M+P'),
    #     'petri32': ('S27-D', 0,  'M+P'),
    # },
}

MOYENNE = False   # True -> une valeur moyennée par pétri (lecteur_données_moy)
                  # False -> 3 spectres par pétri, un par zone (z1/z2/z3)
N_PCA = 8         # nombre de composantes PCA utilisées comme entrée du LDA


# ────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ────────────────────────────────────────────────────────────────────────────
def charger_spectres(config, moyenne=False):
    """Charge tous les spectres et construit les étiquettes associées.

    Si moyenne=False, chaque échantillon donne 3 spectres (z1, z2, z3) qui
    partagent le même identifiant de souris -> ils resteront groupés lors
    de la validation croisée (LeaveOneGroupOut par souris).
    """
    spectres, etiquettes = [], []
    w = None

    for batch, petris in config.items():
        for petri, (echantillon, dose, type_) in petris.items():
            if moyenne:
                a_lire = [(None, lecteur_données_moy(batch, petri))]
            else:
                a_lire = [(z, lecteur_données_zones(batch, petri, z)) for z in ['z1', 'z2', 'z3']]

            for zone, liste_fichiers in a_lire:
                if not liste_fichiers:
                    continue

                w_local, i = traiter_acquisitions_gellose(liste_fichiers)
                if w_local is None or i is None:
                    continue
                if not np.isfinite(i).all():
                    print(f"NaN/Inf : {echantillon} {zone or ''}, {petri}, {batch} — ignoré")
                    continue

                w = w_local
                suffixe = f"_{zone}" if zone else ""
                spectres.append(i)
                etiquettes.append(f"{echantillon}{suffixe}_{dose}{type_}")

    return np.array(spectres), etiquettes, w


def parser_etiquettes(etiquettes):
    """Extrait échantillon, zone, dose, sexe, traitement, id souris depuis les étiquettes.

    Gère les deux formats possibles :
      - "S48-G_z1_45FNT"  (3 segments, moyenne=False)
      - "S48-G_45FNT"     (2 segments, moyenne=True)
    """
    echantillons, zones, doses, sexes, traitements, souris_id = [], [], [], [], [], []

    for e in etiquettes:
        parts = e.split('_')
        echantillon = parts[0]
        zone = parts[1] if len(parts) == 3 else None
        reste = parts[-1]

        m = re.match(r'(\d+)([A-Z])(.*)', reste)
        dose, sexe, traitement = int(m.group(1)), m.group(2), m.group(3)
        mouse = echantillon.split('-')[0]   # "S48-G" -> "S48"

        echantillons.append(echantillon)
        zones.append(zone)
        doses.append(dose)
        sexes.append(sexe)
        traitements.append(traitement)
        souris_id.append(mouse)

    return (
        np.array(echantillons), np.array(zones), np.array(doses),
        np.array(sexes), np.array(traitements), np.array(souris_id),
    )


# ────────────────────────────────────────────────────────────────────────────
# OUTILS D'ÉVALUATION
# ────────────────────────────────────────────────────────────────────────────
def evaluer_lda(X_pca, y, groupes, titre):
    """LDA + validation croisée LeaveOneGroupOut (par souris) + rapport de classification."""
    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, groups=groupes, cv=logo)

    ba = balanced_accuracy_score(y, y_pred)
    print(f"── {titre} ──")
    print(classification_report(y, y_pred))
    print(f"Balanced accuracy : {ba:.1%}\n")

    return y_pred, ba


def test_permutation(X_pca, y, groupes, n_permutations=1000, seed=42):
    """Test de permutation : mélange les étiquettes AU NIVEAU DE LA SOURIS (pas du spectre)."""
    logo = LeaveOneGroupOut()
    y_pred_obs = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, groups=groupes, cv=logo)
    score_observe = balanced_accuracy_score(y, y_pred_obs)

    souris_uniques = np.unique(groupes)
    label_par_souris = {s: y[groupes == s][0] for s in souris_uniques}

    rng = np.random.default_rng(seed)
    scores_permutes = []
    for _ in range(n_permutations):
        labels_melanges = list(label_par_souris.values())
        rng.shuffle(labels_melanges)
        mapping = dict(zip(souris_uniques, labels_melanges))
        y_permute = np.array([mapping[s] for s in groupes])

        y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y_permute, groups=groupes, cv=logo)
        scores_permutes.append(balanced_accuracy_score(y_permute, y_pred))

    scores_permutes = np.array(scores_permutes)
    p_value = (scores_permutes >= score_observe).mean()

    n = len(souris_uniques)
    print(f"Score observé (vrai étiquetage) : {score_observe:.1%}")
    print(f"Score moyen sous permutation (hasard) : {scores_permutes.mean():.1%}")
    print(f"Valeur p empirique : {p_value:.3f}")
    print(f"(Rappel : avec seulement {n} souris, la résolution empirique de p "
          f"dépend du nombre de permutations distinctes possibles — ici {n_permutations} tirages.)")

    return score_observe, scores_permutes, p_value


# ────────────────────────────────────────────────────────────────────────────
# VISUALISATION LDA 1D (projection sur LD1 + histogramme)
# ────────────────────────────────────────────────────────────────────────────
def visualiser_lda_1d(X_lda, etiquettes_points, color_by, color_map, titre,
                       marker_by=None, marker_map=None,
                       legend_color_title="Groupe", legend_marker_title=None):
    n = X_lda.shape[0]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=n)

    fig, (ax_scatter, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True
    )

    for idx in range(n):
        color = color_map[color_by[idx]]
        marker = marker_map[marker_by[idx]] if marker_by is not None else 'o'
        ax_scatter.scatter(X_lda[idx, 0], jitter[idx], color=color, marker=marker, s=50, edgecolors='none')
        ax_scatter.annotate(
            etiquettes_points[idx], xy=(X_lda[idx, 0], jitter[idx]),
            xytext=(3, 3), textcoords='offset points', fontsize=5, color='black', alpha=0.7,
        )

    ax_scatter.set_ylabel("(jitter vertical, sans signification)")
    ax_scatter.axvline(0, color='grey', lw=0.5)
    ax_scatter.set_yticks([])
    ax_scatter.set_title(titre)

    handles_color = [mpatches.Patch(color=c, label=str(k)) for k, c in color_map.items()]
    legend_color = ax_scatter.legend(
        handles=handles_color, title=legend_color_title,
        bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
    )
    ax_scatter.add_artist(legend_color)

    if marker_by is not None:
        handles_marker = [
            Line2D([0], [0], marker=m, color='grey', linestyle='', markersize=8, label=str(k))
            for k, m in marker_map.items()
        ]
        ax_scatter.legend(
            handles=handles_marker, title=legend_marker_title,
            bbox_to_anchor=(1.02, 0.55), loc='upper left', fontsize=8,
        )

    for k, color in color_map.items():
        mask = color_by == k
        ax_hist.hist(X_lda[mask, 0], bins=15, color=color, alpha=0.5, label=str(k))

    ax_hist.set_xlabel("LD1")
    ax_hist.set_ylabel("Nombre de spectres")
    ax_hist.axvline(0, color='grey', lw=0.5)

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
X, etiquettes, w = charger_spectres(CONFIG, moyenne=MOYENNE)
echantillons, zones, doses, sexes, traitements, souris_id = parser_etiquettes(etiquettes)

groupes_dose4 = np.array([f"{d}gy_{t}" for d, t in zip(doses, traitements)])  # 4 classes combinées

print("Répartition des groupes (4 modalités) :")
for g in np.unique(groupes_dose4):
    m = groupes_dose4 == g
    print(f"  {g} : {m.sum()} spectres, {len(np.unique(souris_id[m]))} souris")
print()

X_pca = PCA(n_components=N_PCA).fit_transform(X)

# ── 1) Dose seule (0gy vs 45gy) ───────────────────────────────────────────────
groupes_dose = np.array([f"{d}gy" for d in doses])
y_pred_dose, ba_dose = evaluer_lda(X_pca, groupes_dose, souris_id, "DOSE SEULE (0gy vs 45gy)")

# ── 2) Traitement seul (NT vs +P) ─────────────────────────────────────────────
y_pred_trt, ba_trt = evaluer_lda(X_pca, traitements, souris_id, "TRAITEMENT SEUL (NT vs +P)")

# ── 3) 4 modalités combinées (dose x traitement) ──────────────────────────────
y_pred_4, ba_4 = evaluer_lda(X_pca, groupes_dose4, souris_id, "4 MODALITÉS (dose x traitement)")

# ── Matrices de confusion côte à côte ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
ConfusionMatrixDisplay.from_predictions(groupes_dose, y_pred_dose, ax=axes[0], colorbar=False, normalize='true')
axes[0].set_title("Dose seule")
ConfusionMatrixDisplay.from_predictions(traitements, y_pred_trt, ax=axes[1], colorbar=False, normalize='true')
axes[1].set_title("Traitement seul")
ConfusionMatrixDisplay.from_predictions(groupes_dose4, y_pred_4, ax=axes[2], colorbar=False, normalize='true')
axes[2].set_title("4 modalités")
plt.tight_layout()
plt.show()

# ── LDA final "dose seule" projeté en 1D (LD1), sur toutes les données ───────
color_map_dose = {0: 'blue', 45: 'red'}
marker_map_trt = {'NT': 'o', '+P': 'D'}
etiquettes_points = [f"{echantillons[i]}{sexes[i]}" for i in range(len(etiquettes))]

lda_dose = LinearDiscriminantAnalysis()
X_lda_dose = lda_dose.fit_transform(X_pca, groupes_dose)
print(f"LD1 (dose) — capacité de séparation (eigenvalue) : {lda_dose.explained_variance_ratio_[0]:.1%}")

visualiser_lda_1d(
    X_lda_dose, etiquettes_points, color_by=doses, color_map=color_map_dose,
    titre="LDA — Dose seule (0gy vs 45gy)",
    marker_by=traitements, marker_map=marker_map_trt,
    legend_color_title="Dose", legend_marker_title="Traitement",
)

# ── LDA "4 modalités" projeté en 2D (LD1 vs LD2) ──────────────────────────────
lda4 = LinearDiscriminantAnalysis()
X_lda4 = lda4.fit_transform(X_pca, groupes_dose4)   # jusqu'à 3 axes LD pour 4 classes

fig, ax = plt.subplots(figsize=(8, 7))
for idx in range(len(etiquettes)):
    ax.scatter(
        X_lda4[idx, 0], X_lda4[idx, 1],
        color=color_map_dose[doses[idx]], marker=marker_map_trt[traitements[idx]],
        s=50, edgecolors='none',
    )
    ax.annotate(etiquettes_points[idx], xy=(X_lda4[idx, 0], X_lda4[idx, 1]),
                xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)

ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_title("LDA — 4 modalités (dose x traitement)")
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)

handles_dose = [mpatches.Patch(color=c, label=f"{d}gy") for d, c in color_map_dose.items()]
legend1 = ax.legend(handles=handles_dose, title="Dose", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.add_artist(legend1)
handles_trt = [Line2D([0], [0], marker=m, color='grey', linestyle='', markersize=8, label=t)
               for t, m in marker_map_trt.items()]
ax.legend(handles=handles_trt, title="Traitement", bbox_to_anchor=(1.02, 0.55), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# ── Sous-ensemble 45gy seulement : NT vs +P ───────────────────────────────────
mask_45 = doses == 45
souris_45 = souris_id[mask_45]
etiquettes_points_45 = [etiquettes_points[i] for i in range(len(etiquettes)) if mask_45[i]]

X_pca_45 = PCA(n_components=N_PCA).fit_transform(X[mask_45])
y_pred_45, ba_45 = evaluer_lda(X_pca_45, traitements[mask_45], souris_45, "45gy : NT vs +P")

fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(traitements[mask_45], y_pred_45, ax=ax_cm, colorbar=False, normalize='true')
ax_cm.set_title("45gy : NT vs +P")
plt.tight_layout()
plt.show()

pca_45 = PCA(n_components=N_PCA).fit(X[mask_45])
X_pca_45_full = pca_45.transform(X[mask_45])
lda_45 = LinearDiscriminantAnalysis()
X_lda_45 = lda_45.fit_transform(X_pca_45_full, traitements[mask_45])

color_map_trt_only = {'NT': 'blue', '+P': 'orange'}
visualiser_lda_1d(
    X_lda_45, etiquettes_points_45, color_by=traitements[mask_45], color_map=color_map_trt_only,
    titre="LDA — 45gy : NT vs +P", legend_color_title="Traitement",
)

# ── Spectre discriminant LD1 (45gy, NT vs +P) — quelles régions séparent ─────
discriminant_spectrum = pca_45.components_.T @ lda_45.scalings_[:, 0]

fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
ax_spec.plot(w, discriminant_spectrum, color='darkgreen')
ax_spec.axhline(0, color='grey', lw=0.5)
ax_spec.set_xlabel("Raman shift (cm$^{-1}$)")
ax_spec.set_ylabel("Poids LD1")
ax_spec.set_title("Spectre discriminant LD1 — régions qui séparent NT vs +P (45gy)")
plt.tight_layout()
plt.show()

# ── Test de permutation : traitement seul, sur toutes les données ────────────
test_permutation(X_pca, traitements, souris_id, n_permutations=1000)