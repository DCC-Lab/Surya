from extract_jour2 import traiter_acquisitions_gellose, traiter_acquisitions_verre, extraire_fichiers_j2_fixe, extraire_fichiers_j2_frais
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
jour = 'jour_2'

for support, petris in config.items():
    for petri, (dose, souris_data) in petris.items():
        for souris, zones in souris_data.items():
            if 'fixe' in support:
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