from extracteur_donnees import traiter_acquisitions_gellose, traiter_acquisitions_verre, extraire_fichiers_jour_0, extraire_fichiers_j2_fixe, extraire_fichiers_jour_2,  extraire_fichiers_jour_4, extraire_fichiers_jours_8_11, extraire_fichiers_jour8_frais
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np


config = {

    #'jour0': {
    #    'petri2': ('0gy', {
    #        'souris4': {'echantillon1': ['zone1','zone2','zone3']},
    #        'souris5': {'echantillon1': ['zone1','zone2','zone3']},
    #    }),
    #    'petri3': ('80gy', {
    #        'souris4': {'echantillon1': ['zone2','zone3']},
    #    }),
    #},
    
    'jour_2': {
        #'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3']}),
    },
    'jour4': {
        #'petri1': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri2': ('80gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        },
    'jour_8': {
        #'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
    },  
    'jour_11': {
        #'petri3': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri4': ('80gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
    },
}

extracteur = {
    'jour0':   extraire_fichiers_jour_0,
    'jour_2':   extraire_fichiers_j2_fixe,
    'jour4':   extraire_fichiers_jour_4,
    'jour_8':  extraire_fichiers_jours_8_11,
    'jour_11': extraire_fichiers_jours_8_11,
}

spectres = []
etiquettes = []

for jour, petris in config.items():
    for petri, (dose, souris_data) in petris.items():
        for souris, contenu in souris_data.items():

            if jour == 'jour0':
                # ✅ contenu est un dict {echantillon: [zones]}
                for echantillon, zones in contenu.items():
                    for zone in zones:
                        liste_fichiers = extraire_fichiers_jour_0(jour, petri, souris, echantillon, zone)
                        if not liste_fichiers:
                            continue

                        w, i = traiter_acquisitions_verre(liste_fichiers)  # ou gelose, selon le jour0

                        if w is None or i is None:
                            continue
                        if not np.isfinite(i).all():
                            print(f"NaN/Inf : {souris} {echantillon} {zone}, {petri}, {jour} — ignoré")
                            continue


                        spectres.append(i)
                        etiquettes.append(f"{souris}-{echantillon}-{zone}-{jour}-{dose}")

            elif jour == 'jour_2':
                # ✅ structure normale : contenu est une liste de zones
                zones = contenu
                for zone in zones:
                    liste_fichiers = extracteur[jour]('verre', jour, petri, souris, zone)
                    if not liste_fichiers:
                        continue

                    w, i = traiter_acquisitions_verre(liste_fichiers)

                    if w is None or i is None:
                        continue
                    if not np.isfinite(i).all():
                        print(f"NaN/Inf : {souris} {zone}, {petri}, {jour} — ignoré")
                        continue

                    spectres.append(i)
                    etiquettes.append(f"{souris}-{zone}-{jour}-{dose}")               

            else:
                # ✅ structure normale : contenu est une liste de zones
                zones = contenu
                for zone in zones:
                    liste_fichiers = extracteur[jour](jour, petri, souris, zone)
                    if not liste_fichiers:
                        continue

                    w, i = traiter_acquisitions_verre(liste_fichiers)

                    if w is None or i is None:
                        continue
                    if not np.isfinite(i).all():
                        print(f"NaN/Inf : {souris} {zone}, {petri}, {jour} — ignoré")
                        continue

                    spectres.append(i)
                    etiquettes.append(f"{souris}-{zone}-{jour}-{dose}")


X = np.array(spectres)

# ─────────────────────────────────────────────
# ANALYSE DE LA PCA
# ─────────────────────────────────────────────


# ── 2. Standardiser X (recommandé pour les spectres) ─────────────────────────
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# ── 3. PCA → 3 composantes ───────────────────────────────────────────────────
#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X_scaled)


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
# jour0           : "souris1-echantillon1-zone1-jour0-0gy"  → 5 segments
# autres jours    : "souris1-zone1-jour2-0gy"                → 4 segments

doses  = [e.split('-')[-1] for e in etiquettes]  # toujours le dernier, OK
souris = [e.split('-')[0]  for e in etiquettes]  # toujours le premier, OK

# Pour zone et jour, il faut gérer les deux cas selon le nombre de segments
zones  = []
jours  = []
for e in etiquettes:
    parts = e.split('-')
    if len(parts) == 5:  # jour0 avec échantillon
        zones.append(parts[2])
        jours.append(parts[3])
    else:  # structure normale
        zones.append(parts[1])
        jours.append(parts[2])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (1, 2)]):
    for idx in range(len(etiquettes)):
        dose  = doses[idx]
        jour  = jours[idx]
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

        # Étiquette : jour abrégé + numéro souris + zone
        num_souris = s.replace('souris', '')
        #num_zone   = zone.replace('zone', 'z')
        jour_court = jour.replace('jour_', 'j').replace('jour', 'j')  # jour4→j4, jour_8→j8
        etiquette_point = f"j{jour_court[-1] if '_' not in jour else jour_court[1:]}"

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

