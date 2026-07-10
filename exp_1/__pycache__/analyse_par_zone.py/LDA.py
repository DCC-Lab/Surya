"""
LDA — jour_2, 4 doses (0gy / 45gy / 60gy / 80gy), sans le groupe '45gy + P'.

Premier jour où les 4 doses coexistent (jour0 n'a que du 0gy, avant irradiation).
Attention : une même souris peut contribuer des spectres à plusieurs doses
(ex. souris1 en 0gy ET en 45gy) — le regroupement par souris dans
LeaveOneGroupOut reste néanmoins valide : quand une souris est laissée de côté,
TOUS ses spectres (peu importe la dose) sortent du jeu d'entraînement.
"""

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

from extract_zone import traiter_acquisitions_verre, extraire_fichiers_j2_fixe


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — jour_2, 45gy + P retiré
# ────────────────────────────────────────────────────────────────────────────
CONFIG_JOUR2 = {
    'petri1': ('0gy',  {'souris1': ['zone1'],
                        'souris2': ['zone1', 'zone2'],
                        'souris3': ['zone1', 'zone2', 'zone3']}),
    'petri2': ('45gy', {'souris1': ['zone1', 'zone2'],
                        'souris2': ['zone1', 'zone2', 'zone3']}),
    # 'petri3': ('45gy + P', ...)  -> exclu pour cette première passe
    'petri4': ('60gy', {'souris4': ['zone1', 'zone2', 'zone3'],
                        'souris5': ['zone1', 'zone2', 'zone3']}),
    'petri5': ('80gy', {'souris4': ['zone1', 'zone2', 'zone3']}),
}

N_PCA = 4  # peu de souris distinctes (5) -> peu de composantes pour éviter le surapprentissage


# ────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ────────────────────────────────────────────────────────────────────────────
def charger_jour2(config):
    spectres, doses, souris_id, zones_list = [], [], [], []
    w = None

    for petri, (dose, souris_data) in config.items():
        for souris, zones in souris_data.items():
            for zone in zones:
                liste_fichiers = extraire_fichiers_j2_fixe('verre', 'jour_2', petri, souris, zone)
                if not liste_fichiers:
                    continue

                w_local, i = traiter_acquisitions_verre(liste_fichiers)
                if w_local is None or i is None:
                    continue
                if not np.isfinite(i).all():
                    print(f"NaN/Inf : {souris} {zone}, {petri}, jour_2 — ignoré")
                    continue

                w = w_local
                spectres.append(i)
                doses.append(dose)
                souris_id.append(souris)
                zones_list.append(zone)

    return np.array(spectres), np.array(doses), np.array(souris_id), np.array(zones_list), w


X, doses, souris_id, zones_list, w = charger_jour2(CONFIG_JOUR2)

print("Répartition des groupes (jour_2, sans 45gy + P) :")
for d in np.unique(doses):
    m = doses == d
    print(f"  {d} : {m.sum()} spectres, {len(np.unique(souris_id[m]))} souris "
          f"({', '.join(sorted(np.unique(souris_id[m])))})")
print()


# ────────────────────────────────────────────────────────────────────────────
# PCA + LDA (4 classes) avec validation croisée LeaveOneGroupOut par souris
# ────────────────────────────────────────────────────────────────────────────
X_pca = PCA(n_components=N_PCA).fit_transform(X)

logo = LeaveOneGroupOut()
y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, doses, groups=souris_id, cv=logo)

ba = balanced_accuracy_score(doses, y_pred)
print("── jour_2 : 4 doses (0gy / 45gy / 60gy / 80gy) ──")
print(classification_report(doses, y_pred))
print(f"Balanced accuracy : {ba:.1%}\n")

fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(doses, y_pred, ax=ax_cm, colorbar=False, normalize='true')
ax_cm.set_title("jour_2 — 4 doses")
plt.tight_layout()
plt.show()


# ────────────────────────────────────────────────────────────────────────────
# LDA final entraîné sur toutes les données (projection LD1 vs LD2)
# ────────────────────────────────────────────────────────────────────────────
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_pca, doses)   # jusqu'à 3 axes LD pour 4 classes

print("Capacité de séparation par axe LD :")
for i, v in enumerate(lda.explained_variance_ratio_):
    print(f"  LD{i+1} : {v:.1%}")

color_map = {
    '0gy':  'blue',
    '45gy': 'green',
    '60gy': 'red',
    '80gy': 'purple',
}
marker_map = {
    'souris1': '^',
    'souris2': 's',
    'souris3': 'o',
    'souris4': 'D',
    'souris5': 'P',
}

fig, ax = plt.subplots(figsize=(8, 7))
for idx in range(len(doses)):
    ax.scatter(
        X_lda[idx, 0], X_lda[idx, 1],
        color=color_map[doses[idx]], marker=marker_map[souris_id[idx]],
        s=60, edgecolors='none',
    )
    ax.annotate(zones_list[idx], xy=(X_lda[idx, 0], X_lda[idx, 1]),
                xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)

ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%})")
ax.set_title("LDA — jour_2 : 4 doses")
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)

handles_dose = [mpatches.Patch(color=c, label=d) for d, c in color_map.items()]
legend1 = ax.legend(handles=handles_dose, title="Dose", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.add_artist(legend1)
handles_souris = [Line2D([0], [0], marker=m, color='grey', linestyle='', markersize=8, label=s)
                  for s, m in marker_map.items()]
ax.legend(handles=handles_souris, title="Souris", bbox_to_anchor=(1.02, 0.55), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()


# ────────────────────────────────────────────────────────────────────────────
# Test de permutation (mélange au niveau de la souris)
# ────────────────────────────────────────────────────────────────────────────
def test_permutation(X_pca, y, groupes, n_permutations=1000, seed=42):
    logo = LeaveOneGroupOut()
    y_pred_obs = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, groups=groupes, cv=logo)
    score_observe = balanced_accuracy_score(y, y_pred_obs)

    souris_uniques = np.unique(groupes)
    # attention : ici une souris peut porter plusieurs doses (ses différents spectres),
    # donc on permute les étiquettes spectre par spectre à l'intérieur de chaque souris
    # n'aurait pas de sens ; on permute plutôt l'assignation dose<->souris globalement
    # en conservant la structure (nombre de spectres par souris x dose) — version simple
    # ci-dessous : permutation des étiquettes de dose entre souris-dose "blocs".
    blocs = list(zip(groupes, y))
    uniq_blocs = sorted(set(blocs))

    rng = np.random.default_rng(seed)
    scores_permutes = []
    for _ in range(n_permutations):
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)
        y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y_shuffled, groups=groupes, cv=logo)
        scores_permutes.append(balanced_accuracy_score(y_shuffled, y_pred))

    scores_permutes = np.array(scores_permutes)
    p_value = (scores_permutes >= score_observe).mean()

    print(f"Score observé (vrai étiquetage) : {score_observe:.1%}")
    print(f"Score moyen sous permutation (hasard) : {scores_permutes.mean():.1%}")
    print(f"Valeur p empirique : {p_value:.3f}")

    return score_observe, scores_permutes, p_value


test_permutation(X_pca, doses, souris_id, n_permutations=1000)