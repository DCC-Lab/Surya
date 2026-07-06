from extract_data import traiter_acquisitions_gellose, traiter_acquisitions_verre, lecteur_fichier_j0, lecteur_fichier_j2, lecteur_fichier_j4, lecteur_fichier_j8_j11
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
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
    'jour0': {
        'petri1': ('0gy', {
            'souris1': ['echantillon1', 'echantillon2'],
            'souris2': ['ecantillon1', 'echantillon2'],
            'souris3': ['echantillon1']}),
            
        'petri2': ('0gy', {
            'souris4': ['echantillon1'],
            'souris5': ['echantillon1'],
        }),
        'petri3': ('80gy', {
            'souris4': ['echantillon1'],
        }),
    },
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
    for petri, (dose, souris_data) in petris.items():
        if jour == 'jour0':
            for souris, echantillons in souris_data.items():
                for echantillon in echantillons:
                    liste_fichiers = lecteur_fichier_j0(jour, petri, souris, echantillon)
                    if not liste_fichiers:
                        continue
                    w, i = traiter_acquisitions_verre(liste_fichiers)
                    if w is None or i is None:
                        continue
                    if not np.isfinite(i).all():
                        print(f"NaN/Inf : {souris} {echantillon}, {petri}, {jour} — ignoré")
                        continue
                    spectres.append(i)
                    etiquettes.append(f"{souris}-{jour}-{dose}")
            continue
        else:

            souris_valides = souris_data        
            
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
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# ── 3. PCA → 3 composantes ───────────────────────────────────────────────────
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

print("Variance expliquée par chaque composante :")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1} : {v:.1%}")
print(f"  Total : {sum(pca.explained_variance_ratio_):.1%}")

# ── 4. Plot 3D ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Définir les mappings ──────────────────────────────────────────────────────
color_map = {
    '0gy':      'blue',
    '45gy':     'green',
    '45gy + P': 'orange',
    '60gy':     'red',
    '80gy':     'purple',
}

marker_map = {
    'souris1': '^',
    'souris2': 's',
    'souris3': 'o',
    'souris4': 'D',
    'souris5': 'P',
}

def get_marker(s):
    for cle, marker in marker_map.items():
        if s.startswith(cle):
            return marker
    return 'x'

# ── Extraire dose/souris/jour depuis les étiquettes ──────────────────────────
doses  = [e.split('-')[-1] for e in etiquettes]
souris = [e.split('-')[0]  for e in etiquettes]
jours  = [e.split('-')[1]  for e in etiquettes]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (1, 2)]):
    for idx in range(len(etiquettes)):
        dose   = doses[idx]
        jour   = jours[idx]
        s      = souris[idx]
        color  = color_map[dose]
        marker = get_marker(s)
        est_replique = s.endswith('_1')

        ax.scatter(
            X_reduced[idx, pc_x],
            X_reduced[idx, pc_y],
            color=color,
            marker=marker,
            s=60,
            edgecolors='black' if est_replique else 'none',
            linewidths=1.2,
        )

        # ── Étiquette selon le cas ────────────────────────────────────────────
        if s in ('souris1', 'souris2') and jour == 'jour_8':
            # souris1 à j8 → "#1" pour la distinguer de souris1_1
            num = s.replace('souris', '')
            etiquette_point = f"#1 \n{jour}"
        elif s in ('souris1_1', 'souris2_1'):
            num = s.replace('souris', '').replace('_1', '')
            etiquette_point = f"#2 \n{jour}"
        else:
            # toutes les autres souris → juste le jour
            etiquette_point = jour

        ax.annotate(
            etiquette_point,
            xy=(X_reduced[idx, pc_x], X_reduced[idx, pc_y]),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=5,
            color='black',
            alpha=0.7,
        )

    ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]:.1%})")
    ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]:.1%})")
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)

# ── Légende ───────────────────────────────────────────────────────────────────
handles_dose = [mpatches.Patch(color=c, label=d) for d, c in color_map.items()]

handles_souris = [
    plt.scatter([], [], marker=m, color='grey', label=s)
    for s, m in marker_map.items()
]

axes[1].legend(
    handles=handles_dose + handles_souris,
    title="Dose et Souris",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=7,
)

plt.suptitle("PCA — Score plots")
plt.tight_layout()
plt.show()












# -1- décale tout pour que le minimum soit 0
X_nmf = X - X.min()   
# -2- applique NMF
nmf = NMF(n_components=3, random_state=0)
X_reduced_nmf = nmf.fit_transform(X_nmf)   # ← pas de StandardScaler ! NMF exige des valeurs >= 0


couleurs = ['blue', 'orange', 'green', 'red', 'purple']

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for idx in range(3):
    # ── Colonne gauche : NMF ──────────────────────────────────────────────────
    axes[idx, 0].plot(w, nmf.components_[idx], color=couleurs[idx])
    axes[idx, 0].set_title(f"NMF — Composante {idx+1}")
    axes[idx, 0].set_xlabel("Raman shift (cm$^-1$)")
    axes[idx, 0].set_ylabel("Loading")
    axes[idx, 0].axhline(0, color='grey', lw=0.5)

    # ── Colonne droite : PCA ──────────────────────────────────────────────────
    axes[idx, 1].plot(w, pca.components_[idx], color=couleurs[idx])
    axes[idx, 1].set_title(f"PCA — PC{idx+1} ({pca.explained_variance_ratio_[idx]:.1%} de variance)")
    axes[idx, 1].set_xlabel("Raman shift (cm$^-1$)")
    axes[idx, 1].set_ylabel("Loading")
    axes[idx, 1].axhline(0, color='grey', lw=0.5)

plt.suptitle("NMF vs PCA — Composantes spectrales")
plt.tight_layout()
plt.show()
