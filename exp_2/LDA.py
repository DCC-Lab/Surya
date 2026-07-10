from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import re
from extract_zone import traiter_acquisitions_gellose, lecteur_données_zones, lecteur_données_moy
from sklearn.preprocessing import StandardScaler


config = {
    'batch#1': {
        'petri1':  ('S48-G', 45, 'FNT'),
        'petri2':  ('S48-D', 0, 'FNT'),
        'petri3':  ('S38-G', 45, 'FNT'),
        'petri4':  ('S38-D', 0, 'FNT'),
        'petri5':  ('S40-G', 45, 'FNT'),
        'petri6':  ('S40-D', 0, 'FNT'),
        'petri7':  ('S47-G', 45, 'FNT'),
        'petri8':  ('S47-D', 0, 'FNT'),
        'petri9':  ('S39-G', 0,  'FNT'),
        'petri10': ('S39-D', 0,  'FNT'),
    },
    'batch#2': {
        'petri11': ('S45-G', 45, 'F+P'),
        'petri12': ('S45-D', 0,  'F+P'),
        'petri13': ('S41-G', 45, 'F+P'),
        'petri14': ('S41-D', 0,  'F+P'),
        'petri15': ('S42-G', 0,  'F+P'),
        'petri16': ('S42-D', 0,  'F+P'),
        'petri17': ('S44-G', 0,  'F+P'),
        'petri18': ('S44-D', 0,  'F+P'),
        'petri19': ('S46-G', 0,  'F+P'),
        'petri20': ('S46-D', 0,  'F+P'),
    },
    'batch#3': {
        'petri21': ('S33-G', 45, 'MNT'),
        'petri22': ('S33-D', 0,  'MNT'),
        'petri23': ('S37-G', 45, 'MNT'),
        'petri24': ('S37-D', 0,  'MNT'),
        'petri25': ('S30-G', 45, 'MNT'),
        'petri26': ('S30-D', 0,  'MNT'),
        'petri27': ('S32-G', 45, 'M+P'),
        'petri28': ('S32-D', 0,  'M+P'),
        'petri29': ('S36-G', 45, 'M+P'),
        'petri30': ('S36-D', 0,  'M+P'),
        'petri31': ('S27-G', 45, 'M+P'),
        'petri32': ('S27-D', 0,  'M+P'),
    },
}

spectres = []
etiquettes = []
moyenné = False

for batch, petris in config.items():
    for petri, (echantillon, dose, type_) in petris.items():
        if moyenné:
            liste_fichiers = lecteur_données_moy(batch, petri)
            if not liste_fichiers:
                continue

            w, i = traiter_acquisitions_gellose(liste_fichiers)

            if w is None or i is None:
                continue
            if not np.isfinite(i).all():
                print(f"NaN/Inf : {echantillon}, {petri}, {batch} — ignoré")
                continue

            spectres.append(i)
            etiquettes.append(f"{echantillon}_{dose}{type_}")            
        else:
            for zone in ['z1', 'z2', 'z3']:
                liste_fichiers = lecteur_données_zones(batch, petri, zone)
                if not liste_fichiers:
                    continue

                w, i = traiter_acquisitions_gellose(liste_fichiers)

                if w is None or i is None:
                    continue
                if not np.isfinite(i).all():
                    print(f"NaN/Inf : {echantillon} {zone}, {petri}, {batch} — ignoré")
                    continue

                spectres.append(i)
                etiquettes.append(f"{echantillon}_{zone}_{dose}{type_}")

X = np.array(spectres)

# ── Parsing des étiquettes (comme avant) ──────────────────────────────────────
echantillons = []
zones        = []
doses        = []
sexes        = []
traitements  = []
souris_id    = []   # identifiant unique de la souris (sans -G/-D)

for e in etiquettes:
    parts = e.split('_')
    echantillon = parts[0]           # "S48-G"
    zone        = parts[1]
    reste       = parts[2]

    m = re.match(r'(\d+)([A-Z])(.*)', reste)
    dose       = int(m.group(1))
    sexe       = m.group(2)
    traitement = m.group(3)

    # extraire le code souris sans le côté (-G/-D)
    mouse = echantillon.split('-')[0]   # "S48-G" → "S48"

    echantillons.append(echantillon)
    zones.append(zone)
    doses.append(dose)
    sexes.append(sexe)
    traitements.append(traitement)
    souris_id.append(mouse)

# ── Groupe combiné : dose + traitement (4 classes) ────────────────────────────
groupes = np.array([f"{d}gy_{t}" for d, t in zip(doses, traitements)])
souris_id = np.array(souris_id)

print("Répartition des groupes :")
for g in np.unique(groupes):
    print(f"  {g} : {(groupes == g).sum()} spectres, "
          f"{len(np.unique(souris_id[groupes == g]))} souris")

# ── 1. Scree plot : variance expliquée par composante ─────────────────────────
pca_full = PCA(n_components=min(30, X.shape[0]-1))  # on regarde large pour explorer
pca_full.fit(X)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(range(1, len(pca_full.explained_variance_ratio_)+1),
             pca_full.explained_variance_ratio_, 'o-')
axes[0].set_xlabel("Composante")
axes[0].set_ylabel("Variance expliquée")
axes[0].set_title("Scree plot")
axes[0].axhline(0, color='grey', lw=0.5)

cumul = np.cumsum(pca_full.explained_variance_ratio_)
axes[1].plot(range(1, len(cumul)+1), cumul, 'o-', color='darkorange')
axes[1].axhline(0.95, color='red', linestyle='--', lw=0.8, label='95%')
axes[1].set_xlabel("Nombre de composantes")
axes[1].set_ylabel("Variance cumulée")
axes[1].set_title("Variance cumulée")
axes[1].legend()

plt.tight_layout()
plt.show()

for seuil in [0.90, 0.95, 0.99]:
    if np.any(cumul >= seuil):
        n_needed = np.argmax(cumul >= seuil) + 1
        print(f"  {seuil:.0%} de variance atteinte avec {n_needed} composantes")
    else:
        print(f"  {seuil:.0%} de variance non atteinte avec {len(cumul)} composantes")

# ── 2. Validation croisée : performance du LDA selon n_pca ────────────────────
logo = LeaveOneGroupOut()
valeurs_n_pca = [2, 3, 5, 8, 10, 15, 20, 25, 30]
scores_moyens = []

for n in valeurs_n_pca:
    if n >= X.shape[0]:
        continue
    pca_temp = PCA(n_components=n)
    X_pca_temp = pca_temp.fit_transform(X)

    lda_temp = LinearDiscriminantAnalysis()
    y_pred = cross_val_predict(lda_temp, X_pca_temp, groupes,
                                groups=souris_id, cv=logo)

    score = balanced_accuracy_score(groupes, y_pred)
    scores_moyens.append(score)
    print(f"  n_pca = {n:>2} → balanced accuracy (CV) = {score:.1%}")

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(valeurs_n_pca[:len(scores_moyens)], scores_moyens, 'o-', color='green')
ax2.set_xlabel("Nombre de composantes PCA")
ax2.set_ylabel("Balanced accuracy (validation croisée)")
ax2.set_title("Performance du LDA selon n_pca")
ax2.axhline(1/len(np.unique(groupes)), color='grey', linestyle='--', lw=0.8,
            label=f"Hasard ({1/len(np.unique(groupes)):.0%})")
ax2.legend()
plt.tight_layout()
plt.show()