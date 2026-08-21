"""
Axe discriminant UNIQUE, scalé selon la dose (0/45/60/80 Gy), gélose,
tous les jours poolés ensemble.

Idée : au lieu d'une LDA multiclasse (qui donne min(K-1, p) = 3 axes pour
4 classes nominales, sans aucune notion d'ordre), on traite la dose comme
une variable CONTINUE et on fait une régression PLS avec n_components=1.
PLS trouve la direction w dans l'espace spectral qui maximise la covariance
entre la projection X·w et la dose réelle : c'est exactement l'axe
"intensité du dommage" recherché, et il n'y en a qu'un par construction.

Évaluation : LeaveOneGroupOut par ANIMAL PHYSIQUE (groupe_id = dose_souris),
donc les scores scalaires obtenus via cross_val_predict sont toujours
produits par un modèle qui n'a jamais vu la souris testée (jour0 et jour8
de la même souris tombent dans le même fold, donc pas de fuite).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from scipy.stats import pearsonr

from config import CONFIG2, CONFIG1
from extract_data import extract_jour0, extract_jour2, extract_jour4, extract_jour8_jour11, adjust_spectrum, charger_nocif

extracteur = {
    'jour0':  extract_jour0,
    'jour2':  extract_jour2,
    'jour4':  extract_jour4,
    'jour_8': extract_jour8_jour11,
}


def charger_exp1(config):
    """Charge les spectres gélose, toutes doses/jours confondus.

    groupe_id = identifiant UNIQUE par animal physique = dose + souris,
    car un animal physique n'a qu'une seule dose (donc jour0 et jour8 de
    la même souris tombent toujours dans le même groupe de CV).
    """
    spectres, doses, jours, souris_ids, zones_l, groupe_id = [], [], [], [], [], []

    for jour, petris in config.items():
        for petri, (dose, souris_data) in petris.items():
            for souris, zones in souris_data.items():
                for zone in zones:
                    liste_fichiers = extracteur[jour](petri, souris, zone)
                    print(f"{jour}/{petri}/{souris}/{zone} → {len(liste_fichiers)} fichier(s)")
                    if not liste_fichiers:
                        continue

                    w, i = adjust_spectrum(liste_fichiers, i_nocif=charger_nocif(CONFIG1), retirer_nocif=False)
                    if w is None or i is None:
                        continue
                    if not np.isfinite(i).all():
                        print(f"NaN/Inf : {souris}, {zone}, {jour} — ignoré")
                        continue

                    spectres.append(i)
                    doses.append(dose)
                    jours.append(jour)
                    souris_ids.append(souris)
                    zones_l.append(zone)
                    groupe_id.append(f"{dose}_{souris}")

    return (
        np.array(spectres),
        np.array(doses),
        np.array(jours),
        np.array(souris_ids),
        np.array(zones_l),
        np.array(groupe_id),
        w,
    )

print('charge les spectres')
X, y_dose, jours, souris_id, zones, groupes, w = charger_exp1(CONFIG2)

print("Classes de dose :", np.unique(y_dose, return_counts=True))
print("Nb d'animaux (groupes CV) :", len(np.unique(groupes)))

# ── Dose en variable continue (Gy) ──────────────────────────────────────────
dose_num_map = {'0gy': 0.0, '45gy': 45.0, '60gy': 60.0, '80gy': 80.0}
y_dose_num = np.array([dose_num_map[d] for d in y_dose])

# ── Score PLS out-of-fold, un modèle par souris laissée de côté ────────────
logo = LeaveOneGroupOut()
pls = PLSRegression(n_components=1)

print('fait les prédictions')
y_pred_num = cross_val_predict(pls, X, y_dose_num, groups=groupes, cv=logo).ravel()

r, p = pearsonr(y_dose_num, y_pred_num)
print(f"\nCorrélation score PLS (CV, souris jamais vue) ↔ dose réelle : r={r:.3f}, p={p:.2e}")

# ── Fit final sur tout X pour récupérer l'axe / le "spectre discriminant" ──
print('fit final')
pls_final = PLSRegression(n_components=1)
pls_final.fit(X, y_dose_num)
axe_discriminant = pls_final.x_weights_[:, 0]

print('plotting...')
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(w, axe_discriminant)
ax.axhline(0, color='grey', lw=0.5)
ax.set_xlabel("Nombre d'onde")
ax.set_ylabel("Poids sur l'axe PLS")
ax.set_title("Axe discriminant unique (PLS, dose = variable continue)")
plt.tight_layout()
plt.show()

# ── Projection scalaire out-of-fold, groupée par dose réelle ───────────────
print('projection scalaire')
ordre_classes = ["0gy", "45gy", "60gy", "80gy"]
positions = np.arange(len(ordre_classes))

fig, ax = plt.subplots(figsize=(7, 5))
donnees_par_dose = [y_pred_num[y_dose == d] for d in ordre_classes]
ax.boxplot(donnees_par_dose, positions=positions, widths=0.5, showfliers=False)
for i, d in enumerate(ordre_classes):
    m = y_dose == d
    jitter = (np.random.rand(m.sum()) - 0.5) * 0.3
    ax.scatter(np.full(m.sum(), i) + jitter, y_pred_num[m], alpha=0.7,
               edgecolor='k', zorder=3)

ax.set_xticks(positions)
ax.set_xticklabels(ordre_classes)
ax.set_ylabel("Score PLS prédit (CV, souris jamais vue)")
ax.set_title(f"Projection scalaire sur l'axe unique — r={r:.2f} (p={p:.1e})")
plt.tight_layout()
plt.show()

