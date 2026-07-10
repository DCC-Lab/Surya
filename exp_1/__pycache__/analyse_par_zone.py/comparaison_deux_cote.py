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
import numpy as np

def raman_shift_to_nm(shift_cm1, laser_nm):
    nu_laser = 1e7 / laser_nm          # cm^-1
    nu_scattered = nu_laser - shift_cm1  # Stokes
    return 1e7 / nu_scattered           # nm



config = {
    'jour0': {
        'petri2' :('0gy', {
            'souris4' : {
                'echantillon1': ['zone1', 'zone3'], 
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
                        w_cm, i = traiter_acquisitions_verre(liste_fichiers)  # ou gellose, selon le jour0

                        if w_cm is None or i is None:
                            continue

                        shifts = np.array(w_cm)  # tes valeurs en cm^-1
                        laser_nm = 785
                        nu_laser = 1e7 / laser_nm
                        w = 1e7 / (nu_laser - shifts)

                        if not np.isfinite(i).all():
                            print(f"NaN/Inf : {souris} {echantillon} {zone}, {petri}, {jour} — ignoré")
                            continue

                        spectres.append(i)
                        etiquettes.append(f"{souris}-{echantillon}-{zone}-{jour}-{dose}")    

X = np.array(spectres)        

# --- Regrouper les spectres par échantillon ---
groupes = {'echantillon1': [], 'inversé': []}

for idx, etiquette in enumerate(etiquettes):
    souris, echantillon, zone, jour, dose = etiquette.split('-')
    groupes[echantillon].append(spectres[idx])

# --- Moyenner chaque groupe (3 zones -> 1 spectre moyen) ---
moyenne_echantillon1 = np.mean(np.array(groupes['echantillon1']), axis=0)
moyenne_inverse = np.mean(np.array(groupes['inversé']), axis=0)

# --- Tracer les deux spectres moyens sur le même graphique ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(w, moyenne_echantillon1, color='blue', linewidth=1.2, label='Échantillon 1 (moyenne 3 zones)')
ax.plot(w, moyenne_inverse, color='red', linewidth=1.2, label='Inversé (moyenne 3 zones)')

ax.set_xlabel("Raman shift (cm⁻¹)")
ax.set_ylabel("Intensité")
ax.axhline(0, color='grey', lw=0.3)
ax.legend(title="Côté")
ax.set_title("Comparaison des deux côtés de la peau (moyennes) — Souris 4, Jour 0, 0 Gy")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

couleurs_zone = {
    'zone1': 'blue',
    'zone2': 'orange',
    'zone3': 'green',
}

for idx, etiquette in enumerate(etiquettes):
    souris, echantillon, zone, jour, dose = etiquette.split('-')
    
    # colonne gauche = echantillon1, colonne droite = inversé
    ax = axes[0] if echantillon == 'echantillon1' else axes[1]

    
    
    ax.plot(
        w,
        spectres[idx],
        color=couleurs_zone[zone],
        linewidth=0.6,        # ← plus fin que la moyenne (~1.5)
        label=zone,
    )

axes[0].set_title('Échantillon 1')
axes[1].set_title('Inversé')

for ax in axes:
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensité")
    ax.axhline(0, color='grey', lw=0.3)
    # légende sans doublons
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Zone")

plt.suptitle("Comparaison des deux côtés de la peau — Souris 4, Jour 0, 0 Gy")
plt.tight_layout()
plt.show()