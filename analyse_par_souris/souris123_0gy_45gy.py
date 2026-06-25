from extract_data import traiter_acquisitions_gellose, traiter_acquisitions_verre, lecteur_fichier_j2, lecteur_fichier_j4, lecteur_fichier_j8_j11
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np



# ─────────────────────────────────────────────
# CONSTRUCTION DE LA MATRICE DE DONNÉES
# ─────────────────────────────────────────────
# DIFFÉRENCIER PETRI DE ZONE !!!!!!!!!!!!!!
"""
Construit une matrice de données pour les jours 2,
4, 8 et 11. va avoir la forme :

                    500 nm^-1  ...     3000 nm^-1 
souris1-j2-0gy      0,3       ...       0,5
souris1-j2-45gy     0,2       ...       0,4
        ...
souris5-j11-80gy    0,1       ...       0,3
""" 
# Correspondances pétri → (dose, souris valides)
config = {
    'jour2': {
        'petri1': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri2': ('45gy',     ['souris1', 'souris2']),
    },
    'jour4': {
        'petri3': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri5': ('45gy',     ['souris1', 'souris2']),
    },
    'jour_8': {
        'petri1': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri2': ('45gy',     ['souris1', 'souris2', 'souris3']),
    },
    'jour_11': {
        'petri1': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri2': ('45gy',     ['souris1', 'souris2', 'souris3']),
    },
}

lecteurs = {
    'jour2':  lecteur_fichier_j2,
    'jour4':  lecteur_fichier_j4,
    'jour_8': lecteur_fichier_j8_j11,
    'jour_11':lecteur_fichier_j8_j11,
}

spectres = []
etiquettes = []

for jour, petris in config.items():
    for petri, (dose, souris_valides) in petris.items():
        for souris in souris_valides:
            liste_fichiers = lecteurs[jour](jour, petri, souris)
            if not liste_fichiers:
                continue
            if jour == 'jour2':
                w, i = traiter_acquisitions_gellose(liste_fichiers)
            elif jour == 'jour_8' or jour == 'jour_11' or jour == 'jour4':
                w, i = traiter_acquisitions_verre(liste_fichiers)
            else:
                print(f"⚠️ Jour inconnu : {jour}")
                continue          # ← évite le NameError
            if w is None or i is None:
                continue
            if not np.isfinite(i).all():
                print(f"NaN/Inf : {souris}, {petri}, {jour} — ignoré")
                continue
            spectres.append(i)
            etiquettes.append(f"{souris}-{jour}-{dose}")

# Cas spéciaux souris1.1 et souris2.1 (j8, petri3)
plus_pansement = False
for souris_sp in ['souris1.1', 'souris2.1']:
    souris_label = souris_sp.replace('.', '_')
    liste_fichiers = lecteur_fichier_j8_j11('jour_8', 'petri3', souris_sp)
    if liste_fichiers and plus_pansement:
        w, i = traiter_acquisitions_verre(liste_fichiers)
        if i is not None and np.isfinite(i).all():
            spectres.append(i)
            etiquettes.append(f"{souris_label}-jour_8-45gy + P")


X = np.array(spectres)

# ─────────────────────────────────────────────
# ANALYSE DE LA PCA
# ─────────────────────────────────────────────


# ── 1. Standardiser X (recommandé pour les spectres) ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 2. PCA → 3 composantes ───────────────────────────────────────────────────
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_scaled)

print("Variance expliquée par chaque composante :")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1} : {v:.1%}")
print(f"  Total : {sum(pca.explained_variance_ratio_):.1%}")


