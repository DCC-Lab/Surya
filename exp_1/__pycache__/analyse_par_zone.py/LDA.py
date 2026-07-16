"""
Analyse LDA — expérience 2 (lames de verre), jour_2, 0gy vs 45gy.

Sous-ensemble restreint à petri1 (0gy) et petri2 (45gy) : on retire ici
'45gy + P', '60gy' et '80gy' pour cette première comparaison simple.

ATTENTION : seulement 3 souris distinctes contribuent à ce sous-ensemble
(souris1, souris2, souris3) -> LeaveOneGroupOut ne fait que 3 plis.
Beaucoup plus instable que l'analyse gélose (qui avait ~18-20 souris) :
à interpréter comme un premier coup d'œil exploratoire.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, balanced_accuracy_score, ConfusionMatrixDisplay

from extract_zone import traiter_acquisitions_verre, extraire_fichiers_j2_fixe


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — jour_2, seulement 0gy et 45gy
# ────────────────────────────────────────────────────────────────────────────
CONFIG_JOUR2 = {
    'petri1': ('0gy',  {'souris1': ['zone1'],
                        'souris2': ['zone1', 'zone2'],
                        'souris3': ['zone1', 'zone2', 'zone3']}),
    'petri2': ('45gy', {'souris1': ['zone1', 'zone2'],
                        'souris2': ['zone1', 'zone2', 'zone3']}),
}

N_MAX_COMPOSANTES = 8  # peu d'échantillons ici -> reste prudent, ajuste si besoin


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


# ────────────────────────────────────────────────────────────────────────────
# OUTILS (mêmes fonctions que l'analyse gélose)
# ────────────────────────────────────────────────────────────────────────────
def choisir_n_composantes(X, y, groupes, n_max, titre="Choix du nombre de composantes"):
    """Balaie le nombre de composantes PCA et évalue la balanced accuracy en
    validation croisée LeaveOneGroupOut, pour aider à choisir combien en
    garder avant le LDA final."""
    n_max = min(n_max, X.shape[0] - 2, X.shape[1])  # reste < taille du plus petit jeu d'entraînement
    valeurs_n = list(range(1, max(n_max, 1) + 1))
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
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
X, doses, souris_id, zones_list, w = charger_jour2(CONFIG_JOUR2)

print("Répartition (jour_2, 0gy vs 45gy) :")
for d in np.unique(doses):
    m = doses == d
    print(f"  {d} : {m.sum()} spectres, {len(np.unique(souris_id[m]))} souris "
          f"({', '.join(sorted(np.unique(souris_id[m])))})")
print(f"\nATTENTION : {len(np.unique(souris_id))} souris distinctes au total "
      f"-> LeaveOneGroupOut = {len(np.unique(souris_id))} plis seulement. "
      f"Résultats à interpréter avec prudence.\n")

# ── Sélection du nombre de composantes PCA ────────────────────────────────────
n_pca = choisir_n_composantes(X, doses, souris_id, N_MAX_COMPOSANTES, "Choix N_PCA — jour_2 : 0gy vs 45gy")

# ── LDA — 0gy vs 45gy ──────────────────────────────────────────────────────────
pca = PCA(n_components=n_pca)
X_pca = pca.fit_transform(X)
y_pred, ba = evaluer_lda(X_pca, doses, souris_id, "jour_2 : 0gy vs 45gy")

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_pca, doses)  # 1 axe (2 classes)

# ── Matrice de confusion ───────────────────────────────────────────────────────
fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(doses, y_pred, ax=ax_cm, colorbar=False, normalize='true')
ax_cm.set_title(f"jour_2 : 0gy vs 45gy ({n_pca} composante(s))")
plt.tight_layout()
plt.show()

# ── Projection LD1 (jitter vertical) ──────────────────────────────────────────
color_map_dose = {'0gy': 'blue', '45gy': 'red'}
rng = np.random.default_rng(42)
jitter = rng.uniform(-0.15, 0.15, size=len(doses))

fig, ax = plt.subplots(figsize=(9, 4))
for idx in range(len(doses)):
    ax.scatter(X_lda[idx, 0], jitter[idx], color=color_map_dose[doses[idx]], s=60, edgecolors='none')
    ax.annotate(f"{souris_id[idx]}-{zones_list[idx]}", xy=(X_lda[idx, 0], jitter[idx]),
                xytext=(3, 3), textcoords='offset points', fontsize=6, alpha=0.8)

ax.set_yticks([])
ax.axvline(0, color='grey', lw=0.5)
ax.set_xlabel("LD1")
ax.set_title("LDA — jour_2 : 0gy vs 45gy")
handles = [mpatches.Patch(color=c, label=d) for d, c in color_map_dose.items()]
ax.legend(handles=handles, title="Dose", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# ── Spectre discriminant LD1 ───────────────────────────────────────────────────
discriminant = pca.components_.T @ lda.scalings_[:, 0]

fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
ax_spec.plot(w, discriminant, color='darkred')
ax_spec.axhline(0, color='grey', lw=0.5)
ax_spec.set_xlabel("Raman shift (cm$^{-1}$)")
ax_spec.set_ylabel("Poids LD1")
ax_spec.set_title("Spectre discriminant — jour_2 : 0gy vs 45gy")
plt.tight_layout()
plt.show()