from extracteur_donnees import traiter_acquisitions_gellose, traiter_acquisitions_verre, lecteur_fichier_j2, lecteur_fichier_j4, lecteur_fichier_j8_j11
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
        'petri4': ('60gy',     ['souris4', 'souris5']),
        'petri5': ('80gy',     ['souris4']),
    },
    'jour4': {
        'petri1': ('60gy',     ['souris4', 'souris5']),
        'petri2': ('80gy',     ['souris4', 'souris5']),
    },
    'jour_8': {
        'petri4': ('60gy',     ['souris4', 'souris5']),
        'petri5': ('80gy',     ['souris4', 'souris5']),
    },
    'jour_11': {
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

# ── Palette de couleurs par dose ─────────────────────────────────────────────
doses_uniques = sorted(set(e.split('-')[-1] for e in etiquettes))
palette = {
    '60gy':       '#2196F3',   # bleu
    '80gy':       '#F44336',   # rouge
}
# Couleur de repli pour toute dose non prévue
couleur_defaut = '#9C27B0'

def couleur_dose(etiquette):
    dose = etiquette.split('-')[-1]
    return palette.get(dose, couleur_defaut)

# ── Étiquettes lisibles : "souris1 j2" ────────────────────────────────────────
def label_court(etiquette):
    # Format attendu : "souris1-jour2-0gy"  ou  "souris1_1-jour_8-45gy + P"
    parties = etiquette.split('-')
    souris = parties[0]
    jour   = parties[1].replace('jour_', 'j').replace('jour', 'j')
    return f"{souris} {jour}"

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("ACP des spectres Raman — coloration par dose", fontsize=14, fontweight='bold')

plans = [
    (axes[0], 0, 1, 'PC1', 'PC2'),
    (axes[1], 1, 2, 'PC2', 'PC3'),
]

for ax, idx_x, idx_y, nom_x, nom_y in plans:
    for k, etiq in enumerate(etiquettes):
        x = X_reduced[k, idx_x]
        y = X_reduced[k, idx_y]
        c = couleur_dose(etiq)

        ax.scatter(x, y, color=c, s=70, zorder=3, edgecolors='white', linewidths=0.5)
        ax.annotate(
            label_court(etiq),
            xy=(x, y),
            xytext=(5, 4),
            textcoords='offset points',
            fontsize=7.5,
            color='#333333',
        )

    # Axes et grille
    var_x = pca.explained_variance_ratio_[idx_x]
    var_y = pca.explained_variance_ratio_[idx_y]
    ax.set_xlabel(f"{nom_x} ({var_x:.1%})", fontsize=11)
    ax.set_ylabel(f"{nom_y} ({var_y:.1%})", fontsize=11)
    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.6, linestyle='--')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{nom_x} vs {nom_y}", fontsize=12)

# ── Légende commune ───────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=c, label=dose)
    for dose, c in palette.items()
    if dose in doses_uniques
]
fig.legend(
    handles=patches,
    title='Dose',
    loc='lower center',
    ncol=len(patches),
    bbox_to_anchor=(0.5, -0.04),
    fontsize=10,
    title_fontsize=10,
    frameon=True,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

