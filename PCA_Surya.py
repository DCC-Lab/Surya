from extract_data import traiter_acquisitions_gellose, traiter_acquisitions_verre, lecteur_fichier_j2, lecteur_fichier_j4, lecteur_fichier_j8_j11
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np



# ─────────────────────────────────────────────
# CONSTRUCTION DE LA MATRICE DE DONNÉESss
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
# ANALYSE DE LA PCA
# ─────────────────────────────────────────────


# ── 2. Standardiser X (recommandé pour les spectres) ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. PCA → 3 composantes ───────────────────────────────────────────────────
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_scaled)

print("Variance expliquée par chaque composante :")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1} : {v:.1%}")
print(f"  Total : {sum(pca.explained_variance_ratio_):.1%}")

# ── 4. Plot 3D ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Définir les mappings ───────────────────────────────────────────────────
color_map = {
    '0gy':      'blue',
    '45gy':     'green',
    '45gy + P': 'orange',
    '60gy':     'red',
    '80gy':     'purple',
}

marker_map = {
    'jour2':   '^',   # triangle
    'jour4':   's',   # carré
    'jour_8':  'o',   # cercle
    'jour_11': 'D',   # diamant
}

# ── 2. Extraire dose/souris/jour depuis les étiquettes ───────────────────────
doses  = [e.split('-')[-1] for e in etiquettes]
souris = [e.split('-')[0]  for e in etiquettes]
jours  = [e.split('-')[1]  for e in etiquettes]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
    for idx in range(len(etiquettes)):
        dose   = doses[idx]
        jour   = jours[idx]
        color  = color_map[dose]
        marker = marker_map.get(jour, 'x')   # 'x' si jour inconnu

        ax.scatter(
            X_reduced[idx, pc_x],
            X_reduced[idx, pc_y],
            color=color,
            marker=marker,
            s=60,
        )

        ax.annotate(
            souris[idx],                    # juste le nom, le jour est déjà dans la forme
            xy=(X_reduced[idx, pc_x], X_reduced[idx, pc_y]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=6,
            color=color,
        )

    ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]:.1%})")
    ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]:.1%})")
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)

# ── 4. Légende dose (couleur) ─────────────────────────────────────────────────
handles_dose = [mpatches.Patch(color=c, label=d) for d, c in color_map.items()]

# ── 5. Légende jour (forme) ───────────────────────────────────────────────────
handles_jour = [
    plt.scatter([], [], marker=m, color='grey', label=j)
    for j, m in marker_map.items()
]

axes[1].legend(
    handles=handles_dose + handles_jour,
    title="Dose / Jour",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
)

plt.suptitle("PCA — Score plots")
plt.tight_layout()
plt.show()

# ── 5. Loadings PC1 selon longueur d'onde ────────────────────────────────────
#plt.close('all')



#fig2, ax = plt.subplots(figsize=(10, 4))

#ax.plot(w, pca.components_[0], color='blue')   # components_[0] = PC1
#ax.set_xlabel("Longueur d'onde (nm)")
#ax.set_ylabel("Loading")
#ax.set_title(f"PC1 loading ({pca.explained_variance_ratio_[0]:.1%} de variance)")
#ax.axhline(0, color='grey', lw=0.5)

#plt.tight_layout()
#plt.show()