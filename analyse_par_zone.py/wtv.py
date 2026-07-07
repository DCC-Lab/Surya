from extract_zone import traiter_acquisitions_gellose, traiter_acquisitions_verre, extraire_fichiers_jour_0, extraire_fichiers_j2_fixe, extraire_fichiers_jour_2,  extraire_fichiers_jour_4, extraire_fichiers_jours_8_11, extraire_fichiers_jour8_frais
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np


config = {

    'jour0': {
        'petri1': ('0gy', {
            'souris1': {'echantillon1': ['zone1','zone2','zone3'], 'echantillon2': ['zone1','zone2','zone3']},
            'souris2': {'echantillon1': ['zone1','zone2','zone3'], 'echantillon2': ['zone1','zone2','zone3']},
            'souris3': {'echantillon1': ['zone1','zone2','zone3']},
        }),
        'petri2': ('0gy', {
            'souris4': {'echantillon1': ['zone1','zone2','zone3']},
            'souris5': {'echantillon1': ['zone1','zone2','zone3']},
        }),
        #'petri3': ('80gy', {
        #    'souris4': {'echantillon1': ['zone1','zone2','zone3']},
        #}),
    },
    
    'jour_2': {
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
    
    #'Jour8': {
    #    'petri1': ('0gy',      {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
    #    'petri2': ('45gy',     {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
    #    'petri4': ('60gy',     {'souris4': ['zone1','zone2'], 'souris5': ['zone1','zone2','zone3']}),

    #},    
    'jour_11': {
        'petri1': ('0gy',      {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri2': ('45gy',     {'souris1': ['zone1','zone2','zone3'], 'souris2': ['zone1','zone2','zone3'], 'souris3': ['zone1','zone2','zone3']}),
        'petri3': ('60gy',     {'souris4': ['zone1','zone2','zone3'], 'souris5': ['zone1','zone2','zone3']}),
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
#pca = PCA(n_components=5)
#X_reduced = pca.fit_transform(X_scaled)


# ── 3. PCA → 3 composantes ───────────────────────────────────────────────────
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

print("Variance expliquée par chaque composante :")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1} : {v:.1%}")
print(f"  Total : {sum(pca.explained_variance_ratio_):.1%}")


# -1- décale tout pour que le minimum soit 0
X_nmf = X - X.min()  
# -2- applique NMF
nmf = NMF(n_components=3, random_state=0)
X_reduced_nmf = nmf.fit_transform(X_nmf)   # ← pas de StandardScaler ! NMF exige des valeurs >= 0


from scipy.fft import rfft, rfftfreq
zone = (w >= 2000) & (w <= 2700)
signal = pca.components_[0][zone]
freqs = rfftfreq(len(signal), d=np.mean(np.diff(w[zone])))
spectre_freq = np.abs(rfft(signal - signal.mean()))
plt.plot(freqs, spectre_freq)
plt.title("FFT de la première composante PCA")
plt.show()

