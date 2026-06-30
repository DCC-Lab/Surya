'''
Ici je vais faire la comparaison entre les deux côtés
de la peau de la souris4 jour 0 petri 2 (0 gy)
je vais aussi profiter de cette oportunité pour 
comprendre comment faire la conversion entre des
cm^-1 de raman shift et une vrai longueur d'onde
'''
from extract_zone import traiter_acquisitions_verre, extraire_fichiers_jour_0
import numpy as np
import matplotlib.pyplot as plt


config = {
    'jour0': {
        'petri2' :('0gy', {
            'souris1' : {
                'echantillon1': ['zone1', 'zone2', 'zone3'], 
                'inversé' : ['zone1', 'zone2', 'zone3']
                }
            })
        }
    }


spectres = []
etiquettes = []


for jour, petris in config.items():
    for petri, (dose, souris_data) in petris.items():
        for souris, contenu in souris_data.items():
            for echantillon, zones in contenu.items():
                    for zone in zones:
                        liste_fichiers = extraire_fichiers_jour_0(jour, petri, souris, echantillon, zone)
                        if not liste_fichiers:
                            continue     
                        w, i = traiter_acquisitions_verre(liste_fichiers)  # ou gellose, selon le jour0

                        if w is None or i is None:
                            continue
                        if not np.isfinite(i).all():
                            print(f"NaN/Inf : {souris} {echantillon} {zone}, {petri}, {jour} — ignoré")
                            continue

                        spectres.append(i)
                        etiquettes.append(f"{souris}-{echantillon}-{zone}-{jour}-{dose}")    

X = np.array(spectres)                          

print(X.shape)  # (nombre de spectres, nombre de points)
