"""
Analyse LDA des spectres Raman — gélose.

Deux analyses INDÉPENDANTES :
  - Dose seule       (0gy vs 45gy), peu importe le traitement
  - Traitement seul  (NT vs +P),    peu importe la dose

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
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, balanced_accuracy_score, ConfusionMatrixDisplay

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
        # 'petri9':  ('S39-G', 0,  'FNT'),
        # 'petri10': ('S39-D', 0,  'FNT'),
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

N_MAX_COMPOSANTES = 15  # borne supérieure explorée par le test de sélection


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

                w_local, i, _ = traiter_acquisitions_gellose(liste_fichiers)
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
    """Extrait échantillon, dose, sexe, traitement, id souris depuis les étiquettes.

    Gère les deux formats possibles :
      - "S48-G_z1_45FNT"  (3 segments, moyenne=False)
      - "S48-G_45FNT"     (2 segments, moyenne=True)
    """
    echantillons, doses, sexes, traitements, souris_id = [], [], [], [], []

    for e in etiquettes:
        parts = e.split('_')
        echantillon = parts[0]
        reste = parts[-1]

        m = re.match(r'(\d+)([A-Z])(.*)', reste)
        dose, sexe, traitement = int(m.group(1)), m.group(2), m.group(3)
        mouse = echantillon.split('-')[0]   # "S48-G" -> "S48"

        echantillons.append(echantillon)
        doses.append(dose)
        sexes.append(sexe)
        traitements.append(traitement)
        souris_id.append(mouse)

    return (
        np.array(echantillons), np.array(doses),
        np.array(sexes), np.array(traitements), np.array(souris_id),
    )


# ────────────────────────────────────────────────────────────────────────────
# OUTILS
# ────────────────────────────────────────────────────────────────────────────
def choisir_n_composantes(X, y, groupes, n_max=19, titre="Choix du nombre de composantes"):
    """Balaie le nombre de composantes PCA et évalue la balanced accuracy en
    validation croisée LeaveOneGroupOut, pour aider à choisir combien en
    garder avant le LDA final."""
    n_max = min(n_max, X.shape[0] - 1, X.shape[1])
    valeurs_n = list(range(1, n_max + 1))
    scores = []
    logo = LeaveOneGroupOut()

    for n in valeurs_n:
        pipe = Pipeline([
            ('pca', PCA(n_components=n)),
            ('lda', LinearDiscriminantAnalysis()),
        ])
        y_pred = cross_val_predict(pipe, X, y, groups=groupes, cv=logo)
        scores.append(balanced_accuracy_score(y, y_pred))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valeurs_n, scores, marker='o')
    ax.axhline(0.5, color='grey', lw=0.5, linestyle='--', label='hasard (2 classes)')
    ax.set_xlabel("Nombre de composantes PCA")
    ax.set_ylabel("Balanced accuracy (CV LeaveOneGroupOut)")
    ax.set_title(titre)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    meilleur_n = valeurs_n[int(np.argmax(scores))]
    print(f"{titre} — meilleur score : {max(scores):.1%} avec {meilleur_n} composante(s)\n")
    return meilleur_n


def evaluer_lda(X_pca, y, groupes, titre):
    """LDA + validation croisée LeaveOneGroupOut (par souris) + rapport de classification."""
    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, groups=groupes, cv=logo)

    ba = balanced_accuracy_score(y, y_pred)
    print(f"── {titre} ──")
    print(classification_report(y, y_pred))
    print(f"Balanced accuracy : {ba:.1%}\n")

    return y_pred, ba


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
X, etiquettes, w = charger_spectres(CONFIG, moyenne=MOYENNE)
echantillons, doses, sexes, traitements, souris_id = parser_etiquettes(etiquettes)

groupes_dose = np.array([f"{d}gy" for d in doses])

groupes_4 = np.array([f"{d}gy_{t}" for d, t in zip(doses, traitements)])
# ex: ['0gy_NT', '45gy_NT', '0gy_+P', '45gy_+P', ...]
print(np.unique(groupes_4, return_counts=True))

print("Répartition :")
for g in np.unique(groupes_dose):
    m = groupes_dose == g
    print(f"  {g} : {m.sum()} spectres, {len(np.unique(souris_id[m]))} souris")
for t in np.unique(traitements):
    m = traitements == t
    print(f"  {t} : {m.sum()} spectres, {len(np.unique(souris_id[m]))} souris")
print()

# ── Sélection du nombre de composantes PCA, pour chaque analyse ──────────────
n_pca_dose = choisir_n_composantes(X, groupes_dose, souris_id, N_MAX_COMPOSANTES, "Choix N_PCA — Dose")
n_pca_trt = choisir_n_composantes(X, traitements, souris_id, N_MAX_COMPOSANTES, "Choix N_PCA — Traitement")
n_pca_4 = choisir_n_composantes(X, groupes_4, souris_id, N_MAX_COMPOSANTES, "Choix N_PCA — 4 groupes")

# ── LDA — Dose seule (0gy vs 45gy), peu importe le traitement ────────────────
pca_dose = PCA(n_components=n_pca_dose)
X_pca_dose = pca_dose.fit_transform(X)
y_pred_dose, ba_dose = evaluer_lda(X_pca_dose, groupes_dose, souris_id, "DOSE SEULE (0gy vs 45gy)")

lda_dose = LinearDiscriminantAnalysis()
X_lda_dose = lda_dose.fit_transform(X_pca_dose, groupes_dose)  # 1 axe (2 classes)

# ── LDA — Traitement seul (NT vs +P), peu importe la dose ────────────────────
pca_trt = PCA(n_components=n_pca_trt)
X_pca_trt = pca_trt.fit_transform(X)
y_pred_trt, ba_trt = evaluer_lda(X_pca_trt, traitements, souris_id, "TRAITEMENT SEUL (NT vs +P)")

lda_trt = LinearDiscriminantAnalysis()
X_lda_trt = lda_trt.fit_transform(X_pca_trt, traitements)


# ── LDA ─ dose et traitement (0gy vs 45gy vs NT vs +P) ────────────────────────
pca_4 = PCA(n_components=n_pca_4)
X_pca_4 = pca_4.fit_transform(X)
y_pred_4, ba_4 = evaluer_lda(X_pca_4, groupes_4, souris_id, "4 GROUPES (dose × traitement)")

lda_4 = LinearDiscriminantAnalysis()
X_lda_4 = lda_4.fit_transform(X_pca_4, groupes_4)  # jusqu'à 3 axes (k-1 = 4-1 = 3)

# ── Matrices de confusion côte à côte ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
ConfusionMatrixDisplay.from_predictions(groupes_dose, y_pred_dose, ax=axes[0], colorbar=False, normalize='true')
axes[0].set_title("Dose seule")
ConfusionMatrixDisplay.from_predictions(traitements, y_pred_trt, ax=axes[1], colorbar=False, normalize='true')
axes[1].set_title("Traitement seul")
ConfusionMatrixDisplay.from_predictions( groupes_4, y_pred_4, ax=ax, colorbar=True, normalize='true', xticks_rotation=45,) # ── Matrice de confusion (4 groupes)
axes[2].set_title("Matrice de confusion — 4 groupes (CV LeaveOneGroupOut)")
plt.tight_layout()
plt.show()

# ── Graphique combiné 2D : composante dose (x) vs composante traitement (y) ──
color_map_dose = {0: 'blue', 45: 'red'}
marker_map_trt = {'NT': 'o', '+P': 'D'}
etiquettes_points = [f"{echantillons[i]}{sexes[i]}" for i in range(len(etiquettes))]

fig, ax = plt.subplots(figsize=(8, 7))
for idx in range(len(etiquettes)):
    ax.scatter(
        X_lda_dose[idx, 0], X_lda_trt[idx, 0],
        color=color_map_dose[doses[idx]], marker=marker_map_trt[traitements[idx]],
        s=60, edgecolors='none',
    )
    ax.annotate(etiquettes_points[idx], xy=(X_lda_dose[idx, 0], X_lda_trt[idx, 0]),
                xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)

ax.set_xlabel("LD1 — dose")
ax.set_ylabel("LD1 — traitement")
ax.set_title("Projection combinée : composante dose vs composante traitement")
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

# ── Spectres discriminants (poids LD1) pour chaque analyse ────────────────────
discriminant_dose = pca_dose.components_.T @ lda_dose.scalings_[:, 0]
discriminant_trt = pca_trt.components_.T @ lda_trt.scalings_[:, 0]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(w, discriminant_dose, color='darkred')
axes[0].axhline(0, color='grey', lw=0.5)
axes[0].set_ylabel("Poids LD1 (dose)")
axes[0].set_title("Spectre discriminant — Dose (0gy vs 45gy)")

axes[1].plot(w, discriminant_trt, color='darkgreen')
axes[1].axhline(0, color='grey', lw=0.5)
axes[1].set_ylabel("Poids LD1 (traitement)")
axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
axes[1].set_title("Spectre discriminant — Traitement (NT vs +P)")

plt.tight_layout()
plt.show()