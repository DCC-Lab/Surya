from exp_2.frais.extract_data import traiter_acquisitions_gellose, lecteur_données_zones, lecteur_données_moy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np


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
    #    'petri9':  ('S39-G', 0,  'FNT'),
    #    'petri10': ('S39-D', 0,  'FNT'),
    },
    'batch#2': {
        'petri11': ('S45-G', 45, 'F+P'),
        'petri12': ('S45-D', 0,  'F+P'),
        'petri13': ('S41-G', 45, 'F+P'),
        'petri14': ('S41-D', 0,  'F+P'),
        'petri15': ('S42-G', 45,  'F+P'),
        'petri16': ('S42-D', 0,  'F+P'),
        'petri17': ('S44-G', 45,  'F+P'),
        'petri18': ('S44-D', 0,  'F+P'),
        'petri19': ('S46-G', 45,  'F+P'),
        'petri20': ('S46-D', 0,  'F+P'),
    },
    #'batch#3': {
    #    'petri21': ('S33-G', 45, 'MNT'),
    #    'petri22': ('S33-D', 0,  'MNT'),
    #    'petri23': ('S37-G', 45, 'MNT'),
    #    'petri24': ('S37-D', 0,  'MNT'),
    #    'petri25': ('S30-G', 45, 'MNT'),
    #    'petri26': ('S30-D', 0,  'MNT'),
    #    'petri27': ('S32-G', 45, 'M+P'),
    #    'petri28': ('S32-D', 0,  'M+P'),
    #    'petri29': ('S36-G', 45, 'M+P'),
    #    'petri30': ('S36-D', 0,  'M+P'),
    #    'petri31': ('S27-G', 45, 'M+P'),
    #    'petri32': ('S27-D', 0,  'M+P'),
    #},
}

spectres = []
etiquettes = []
moyenné = True

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

# ─────────────────────────────────────────────
# ANALYSE DE LA PCA
# ─────────────────────────────────────────────
# ── 2. Standardiser X (recommandé pour les spectres) ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
from matplotlib.lines import Line2D
import re

# ── Mapping couleur selon dose ────────────────────────────────────────────────
color_map = {
    0:  'blue',
    45: 'red',
}

# ── Mapping marqueur selon traitement (NT vs +P) ──────────────────────────────
marker_map = {
    'NT': 'o',   # rond
    '+P': 'D',   # losange
}

# ── Parsing des étiquettes : "S48-G_z1_45FNT" → echantillon, zone, dose, sexe, traitement
echantillons = []
zones        = []
doses        = []
sexes        = []
traitements  = []

for e in etiquettes:
    parts = e.split('_')            # ["S48-G", "z1", "45FNT"]
    echantillon = parts[0]
    if moyenné:
        reste       = parts[1]
    else:
        zone        = parts[1]
        reste       = parts[2]          # "45FNT" ou "0F+P"

    m = re.match(r'(\d+)([A-Z])(.*)', reste)
    dose       = int(m.group(1))    # 0 ou 45
    sexe       = m.group(2)         # 'F' ou 'M'
    traitement = m.group(3)         # 'NT' ou '+P'

    echantillons.append(echantillon)
    if moyenné:
        doses.append(dose)
        sexes.append(sexe) 
    else:       
            
        zones.append(zone)
        doses.append(dose)
        sexes.append(sexe)
    traitements.append(traitement)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (1, 2)]):
    for idx in range(len(etiquettes)):
        color  = color_map[doses[idx]]
        marker = marker_map[traitements[idx]]

        ax.scatter(
            X_reduced[idx, pc_x],
            X_reduced[idx, pc_y],
            color=color,
            marker=marker,
            s=50,
            edgecolors='none',
        )

        # Étiquette : code échantillon + sexe, ex. "S32-GM"
        etiquette_point = f"{echantillons[idx]}{sexes[idx]}"

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

# ── Légende : dose (couleur) ──────────────────────────────────────────────────
handles_dose = [mpatches.Patch(color=c, label=f"{d}gy") for d, c in color_map.items()]
legend_dose = axes[1].legend(
    handles=handles_dose,
    title="Dose",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=8,
)
axes[1].add_artist(legend_dose)

# ── Légende : traitement (marqueur) ───────────────────────────────────────────
handles_traitement = [
    Line2D([0], [0], marker=m, color='grey', linestyle='', markersize=8, label=t)
    for t, m in marker_map.items()
]
axes[1].legend(
    handles=handles_traitement,
    title="Traitement",
    bbox_to_anchor=(1.05, 0.6),
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