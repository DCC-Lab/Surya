from extract_jour2 import traiter_acquisitions_gellose, traiter_acquisitions_verre, extraire_fichiers_j2_fixe, extraire_fichiers_j2_frais
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


config = {
    'frais': {
        'petri1': ('0gy',      {'souris1': ['zone1'], 'souris2': ['zone1','zone2'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2'], 'souris2': ['zone1','zone2','zone3']}),
        'petri3': ('45gy + P', {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3']}),
    },

    'fixe sur verre': {
        'petri1': ('0gy',      {'souris1': ['zone1'], 'souris2': ['zone1','zone2'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2'], 'souris2': ['zone1','zone2','zone3']}),
        'petri3': ('45gy + P', {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3']}), 
    }, 


    'fixe sur gelose': {
        'petri1': ('0gy',      {'souris1': ['zone1'], 'souris2': ['zone1','zone2'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2'], 'souris2': ['zone1','zone2','zone3']}),
        'petri3': ('45gy + P', {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3']}),   
    },
}

extracteur = {
    'frais':   extraire_fichiers_j2_frais,
    'fixe sur verre': extraire_fichiers_j2_fixe,
    'fixe sur gelose': extraire_fichiers_j2_fixe,
}

spectres = []
etiquettes = []

for support, petris in config.items():
    for petri, (dose, souris_data) in petris.items():
        for souris, zones in souris_data.items():
            if 'fixe' in support:
                jour = 'jour_2'
                for zone in zones:
                    liste_fichiers = extracteur[support](support.split(' ')[-1], jour, petri, souris, zone)
                    if not liste_fichiers:
                        continue
                    if 'verre' in support:
                        w, i = traiter_acquisitions_verre(liste_fichiers)
                    else:
                        w, i = traiter_acquisitions_gellose(liste_fichiers)

                    if w is None or i is None:
                        continue
                    if not np.isfinite(i).all():
                        print(f"NaN/Inf : {souris} {zone}, {petri}, {jour} — ignoré")
                        continue

                    spectres.append(i)
                    etiquettes.append(f"{souris}-{zone}-{support}-{dose}")
            else:
                jour = 'jour2'
                for zone in zones:
                    liste_fichiers = extracteur[support](jour, petri, souris, zone)
                    if not liste_fichiers:
                        continue

                    w, i = traiter_acquisitions_gellose(liste_fichiers)

                    if w is None or i is None:
                        continue
                    if not np.isfinite(i).all():
                        print(f"NaN/Inf : {souris} {zone}, {petri}, {jour} — ignoré")
                        continue

                    spectres.append(i)
                    etiquettes.append(f"{souris}-{zone}-{support}-{dose}")                




X = np.array(spectres)

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
# jour0           : "souris1-echantillon1-zone1-jour0-0gy"  → 5 segments
# autres jours    : "souris1-zone1-jour2-0gy"                → 4 segments

doses  = [e.split('-')[-1] for e in etiquettes]
souris = [e.split('-')[0]  for e in etiquettes]

zones    = []
supports = []
for e in etiquettes:
    parts = e.split('-')
    if len(parts) == 5:  # ancien format jour0 avec échantillon, si jamais présent
        zones.append(parts[2])
        supports.append(parts[3])
    else:
        zones.append(parts[1])
        supports.append(parts[2])

# Abréviations pour les étiquettes sur le graphique
support_abr = {
    'frais':            'F',
    'fixe sur verre':   'V',
    'fixe sur gelose':  'G',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (1, 2)]):
    for idx in range(len(etiquettes)):
        dose  = doses[idx]
        s     = souris[idx]
        #zone  = zones[idx]
        color = color_map[dose]

        ax.scatter(
            X_reduced[idx, pc_x],
            X_reduced[idx, pc_y],
            color=color,
            marker=get_marker(s),
            s=50,
            edgecolors='none',
        )

        support_txt = supports[idx]
        zone_txt    = zones[idx].replace('zone', 'z')  # "zone1" → "z1"
        etiquette_point = f"{support_abr.get(support_txt, support_txt)}-{zone_txt}"

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

#Légende : seulement les doses
handles_dose = [mpatches.Patch(color=c, label=d) for d, c in color_map.items()]
axes[1].legend(
    handles=handles_dose,
    title="Dose",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=8,
)

handles_souris = [
    Line2D([0], [0], marker=m, color='grey', linestyle='', markersize=8, label=s)
    for s, m in marker_map.items()
]


legend_dose = axes[1].legend(
    handles=handles_dose,
    title="Dose",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=8,
)
axes[1].add_artist(legend_dose)

axes[1].legend(
    handles=handles_souris,
    title="Souris",
    bbox_to_anchor=(1.05, 0.5),
    loc='upper left',
    fontsize=8,
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

plt.suptitle("NMF vs PCA — Composantes spectrales-3025-moyennées-sans scaled")
plt.tight_layout()
plt.show()
