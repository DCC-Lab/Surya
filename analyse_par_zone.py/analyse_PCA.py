from extract_zone import traiter_acquisitions_gellose, traiter_acquisitions_verre, extraire_fichiers_jour_2, extraire_fichiers_jour_4, extraire_fichiers_jours_8_11
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


config = {
    'jour2': {
        'petri1': ('0gy',      {'souris1': ['zone1'], 'souris2': ['zone1','zone2'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3']}),
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
jours  = [e.split('-')[1]  for e in etiquettes]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
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
            etiquette_point = f"#{num}\n{jour}"
        elif s in ('souris1_1', 'souris2_1'):
            # souris1_1 → "#2" (deuxième individu)
            num = s.replace('souris', '').replace('_1', '')
            etiquette_point = f"#{num} bis\n{jour}"
        else:
            # toutes les autres souris → juste le jour
            etiquette_point = jour

        ax.annotate(
            etiquette_point,
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

# ── Légende ───────────────────────────────────────────────────────────────────
handles_dose = [mpatches.Patch(color=c, label=d) for d, c in color_map.items()]

handles_souris = [
    plt.scatter([], [], marker=m, color='grey', label=s)
    for s, m in marker_map.items()
]

handles_replique = [
    plt.scatter([], [], marker='o', color='grey', edgecolors='none',  label='souris originale'),
    plt.scatter([], [], marker='o', color='grey', edgecolors='black', linewidths=1.2, label='souris _1 (réplique)'),
]

axes[1].legend(
    handles=handles_dose + handles_souris + handles_replique,
    title="Dose / Souris / Réplique",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=7,
)

plt.suptitle("PCA — Score plots")
plt.tight_layout()
plt.show()