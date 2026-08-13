from extract_data import traiter_acquisitions_gellose, traiter_acquisitions_verre, lecteur_fichier_j2, lecteur_fichier_j4, lecteur_fichier_j8_j11
from sklearn.decomposition import NMF
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
        'petri3': ('45gy + P', ['souris1', 'souris2', 'souris3']),
        'petri4': ('60gy',     ['souris4', 'souris5']),
        'petri5': ('80gy',     ['souris4']),
    },
    'jour4': {
        'petri1': ('60gy',     ['souris4', 'souris5']),
        'petri2': ('80gy',     ['souris4', 'souris5']),
        'petri3': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri4': ('45gy + P', ['souris1', 'souris2', 'souris3']),
        'petri5': ('45gy',     ['souris1', 'souris2']),
    },
    'jour_8': {
        'petri1': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri2': ('45gy',     ['souris1', 'souris2', 'souris3']),
        'petri3': ('45gy + P', ['souris1', 'souris2']),  # souris1.1 et 2.1 gérés séparément
        'petri4': ('60gy',     ['souris4', 'souris5']),
        'petri5': ('80gy',     ['souris4', 'souris5']),
    },
    'jour_11': {
        'petri1': ('0gy',      ['souris1', 'souris2', 'souris3']),
        'petri2': ('45gy',     ['souris1', 'souris2', 'souris3']),
        'petri3': ('60gy',     ['souris4', 'souris5']),
        'petri4': ('80gy',     ['souris4', 'souris5']),
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
for souris_sp in ['souris1.1', 'souris2.1']:
    souris_label = souris_sp.replace('.', '_')
    liste_fichiers = lecteur_fichier_j8_j11('jour_8', 'petri3', souris_sp)
    if liste_fichiers:
        w, i = traiter_acquisitions_verre(liste_fichiers)
        if i is not None and np.isfinite(i).all():
            spectres.append(i)
            etiquettes.append(f"{souris_label}-jour_8-45gy + P")


X = np.array(spectres)


# ─────────────────────────────────────────────
# NON-NEGATIVE MATRIX FACTORIZATION (NMF)
# ─────────────────────────────────────────────
# -1- décale tout pour que le minimum soit 0
X_nmf = X - X.min()   
# -2- applique NMF
nmf = NMF(n_components=3, random_state=0)
X_reduced_nmf = nmf.fit_transform(X_nmf)   # ← pas de StandardScaler ! NMF exige des valeurs >= 0

# ─────────────────────────────────────────────
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# ─────────────────────────────────────────────
# -1- applique PCA
pca = PCA(n_components=3)
X_reduced_pca = pca.fit_transform(X_nmf)


fig, axes = plt.subplots(3, 2, figsize=(14, 10))
couleurs = ['blue', 'orange', 'green']

# ─────────────────────────────────────────────
# AFFICHAGE NMF VS PCA
# ─────────────────────────────────────────────

for idx in range(3):
    # ── Colonne gauche : NMF ──────────────────────────────────────────────────
    axes[idx, 0].plot(w, nmf.components_[idx], color=couleurs[idx])
    axes[idx, 0].set_title(f"NMF — Composante {idx+1}")
    axes[idx, 0].set_xlabel("Longueur d'onde (nm)")
    axes[idx, 0].set_ylabel("Loading")
    axes[idx, 0].axhline(0, color='grey', lw=0.5)

    # ── Colonne droite : PCA ──────────────────────────────────────────────────
    axes[idx, 1].plot(w, pca.components_[idx], color=couleurs[idx])
    axes[idx, 1].set_title(f"PCA — PC{idx+1} ({pca.explained_variance_ratio_[idx]:.1%} de variance)")
    axes[idx, 1].set_xlabel("Longueur d'onde (nm)")
    axes[idx, 1].set_ylabel("Loading")
    axes[idx, 1].axhline(0, color='grey', lw=0.5)

plt.suptitle("NMF vs PCA — Composantes spectrales")
plt.tight_layout()
plt.show()