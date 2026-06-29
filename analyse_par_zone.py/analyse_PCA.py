from extract_zone import traiter_acquisitions_gellose, traiter_acquisitions_verre, extraire_fichiers_jour_2, extraire_fichiers_jour_4, extraire_fichiers_jours_8_11
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


config = {
    'jour2': {
        'petri1': ('0gy',      {'souris1': ['zone1'], 'souris2': ['zone1','zone2'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2'], 'souris2': ['zone1','zone2','zone3']}),
        'petri3': ('45gy + P', {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3']}),
    },
    'jour4': {
        'petri1': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri2': ('80gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri3': ('0gy',      {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri4': ('45gy + P', {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri5': ('45gy',     {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3']}),
    },
    'jour_8': {
        'petri1': ('0gy',      {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri3': ('45gy + P', {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3']}),
        'petri4': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri5': ('80gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
    },
    'jour_11': {
        'petri1': ('0gy',      {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri3': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
        'petri4': ('80gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
    },
}

extracteur = {
    'jour2':  extraire_fichiers_jour_2,
    'jour4':  extraire_fichiers_jour_4,
    'jour_8': extraire_fichiers_jours_8_11,
    'jour_11':extraire_fichiers_jours_8_11,
}


spectres = []
etiquettes = []

for jour, petris in config.items():
    for petri, (dose, souris_zones) in petris.items():
        for souris, zones in souris_zones.items():
            for zone in zones:
                liste_fichiers = extracteur[jour](jour, petri, souris, zone)
                if not liste_fichiers:
                    continue

                if jour == 'jour2':
                    w, i = traiter_acquisitions_gellose(liste_fichiers)
                elif jour in ('jour4', 'jour_8', 'jour_11'):
                    w, i = traiter_acquisitions_verre(liste_fichiers)
                else:
                    print(f"⚠️ Jour inconnu : {jour}")
                    continue

                if w is None or i is None:
                    continue
                if not np.isfinite(i).all():
                    print(f"NaN/Inf : {souris} {zone}, {petri}, {jour} — ignoré")
                    continue

                spectres.append(i)
                # ✅ L'étiquette inclut maintenant la zone
                etiquettes.append(f"{souris}-{zone}-{jour}-{dose}")

# Cas spéciaux souris1.1 et souris2.1 (j8, petri3)
for souris_sp in ['souris1.1', 'souris2.1']:
    souris_label = souris_sp.replace('.', '_')
    for zone in ['zone1', 'zone2', 'zone3']:
        liste_fichiers = extraire_fichiers_jours_8_11('jour_8', 'petri3', souris_sp, zone)
        if liste_fichiers:
            w, i = traiter_acquisitions_verre(liste_fichiers)
            if i is not None and np.isfinite(i).all():
                spectres.append(i)
                etiquettes.append(f"{souris_label}-{zone}-jour_8-45gy + P")


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
zones  = [e.split('-')[1]  for e in etiquettes]
jours  = [e.split('-')[2]  for e in etiquettes]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
    for idx in range(len(etiquettes)):
        dose  = doses[idx]
        jour  = jours[idx]
        s     = souris[idx]
        zone  = zones[idx]
        color = color_map[dose]

        ax.scatter(
            X_reduced[idx, pc_x],
            X_reduced[idx, pc_y],
            color=color,
            marker='o',
            s=50,
            edgecolors='none',
        )

        # Étiquette : jour abrégé + numéro souris + zone
        num_souris = s.replace('souris', '')
        num_zone   = zone.replace('zone', 'z')
        jour_court = jour.replace('jour_', 'j').replace('jour', 'j')  # jour4→j4, jour_8→j8
        etiquette_point = f"j{jour_court[-1] if '_' not in jour else jour_court[1:]}·s{num_souris}·{num_zone}"

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

# Légende : seulement les doses
handles_dose = [mpatches.Patch(color=c, label=d) for d, c in color_map.items()]
axes[1].legend(
    handles=handles_dose,
    title="Dose",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=8,
)

plt.suptitle("PCA — Score plots")
plt.tight_layout()
plt.show()