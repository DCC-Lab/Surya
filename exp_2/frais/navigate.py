import numpy as np
import matplotlib.pyplot as plt
from extract_data import traiter_acquisitions_gellose, lecteur_données_frais, lecteur_données_fixes, lecteur_données_moy, traiter_acquisitions_verre, extraire_fichiers_jour_0, extraire_fichiers_j2_fixe,  extraire_fichiers_jour_4, extraire_fichiers_jours_8_11



config = {
    'batch#1': {
        'petri1':  ('S48-G', 45, 'FNT'),
        'petri2':  ('S48-D', 0,  'FNT'),
        'petri3':  ('S38-G', 45, 'FNT'),
        'petri4':  ('S38-D', 0,  'FNT'),
        'petri5':  ('S40-G', 45, 'FNT'),
        'petri6':  ('S40-D', 0,  'FNT'),
        'petri7':  ('S47-G', 45, 'FNT'),
        'petri8':  ('S47-D', 0,  'FNT'),
        # 'petri9':  ('S39-G', 0,  'FNT'),
        # 'petri10': ('S39-D', 0,  'FNT'),
    },
    'batch#2': {
        'petri11': ('S45-G', 45, 'F+P'),
        'petri12': ('S45-D', 0,  'F+P'),
        'petri13': ('S41-G', 45, 'F+P'),
        'petri14': ('S41-D', 0,  'F+P'),
        'petri15': ('S42-G', 45, 'F+P'),
        'petri16': ('S42-D', 0,  'F+P'),
        'petri17': ('S44-G', 45, 'F+P'),
        'petri18': ('S44-D', 0,  'F+P'),
        'petri19': ('S46-G', 45, 'F+P'),
        'petri20': ('S46-D', 0,  'F+P'),
    },
     'batch#3': {
         'petri21': ('S33-G', 45, 'MNT'),
         'petri22': ('S33-D', 0,  'MNT'),
         'petri23': ('S37-G', 45, 'MNT'),
         'petri24': ('S37-D', 0,  'MNT'),
         'petri25': ('S30-G', 45, 'MNT'),
         'petri26': ('S30-D', 0,  'MNT'),
         'petri27': ('S32-G', 45, 'M+P'),
         'petri28': ('S32-D', 0,  'M+P'),
         'petri29': ('S36-G', 45, 'M+P'),
         'petri30': ('S36-D', 0,  'M+P'),
         'petri31': ('S27-G', 45, 'M+P'),
         'petri32': ('S27-D', 0,  'M+P'),
     },
    'batch#4': {
         'petri33': ('S29-G', 0,  'MNT'),
         'petri34': ('S29-D', 0,  'MNT'),
         'petri35': ('S31-G', 45, 'MNT'),
         'petri36': ('S31-D', 0,  'MNT'),
         'petri37': ('S34-G', 45, 'M+P'),
         'petri38': ('S34-D', 0,  'M+P'),

     },
}


i_non_irr, ip_non_irr, i_irr, ip_irr =[], [], [], []


for batch, petri in config.items():
    for petri, (echantillon, dose, type_) in petri.items():
        if dose == 0:
            if 'NT' in type_:
                for z in ('z1', 'z2', 'z3'):
                    fichiers = lecteur_données_frais(batch, petri, z)
                    if not fichiers:
                        print(f"⚠ Aucun fichier pour {petri} {z} — ignoré")
                        continue
                    w, i = traiter_acquisitions_gellose(fichiers)
                    i_non_irr.append(i)
            else:
                for z in ('z1', 'z2', 'z3'):
                    fichiers = lecteur_données_frais(batch, petri, z)
                    if not fichiers:
                        print(f"⚠ Aucun fichier pour {petri} {z} — ignoré")
                        continue
                    w, i = traiter_acquisitions_gellose(fichiers)
                    ip_non_irr.append(i)  
        elif dose == 45:
            if 'NT' in type_:
                for z in ('z1', 'z2', 'z3'):
                    fichiers = lecteur_données_frais(batch, petri, z)
                    if not fichiers:
                        print(f"⚠ Aucun fichier pour {petri} {z} — ignoré")
                        continue
                    w, i = traiter_acquisitions_gellose(fichiers)
                    i_irr.append(i)  
            else:
                for z in ('z1', 'z2', 'z3'):
                    fichiers = lecteur_données_frais(batch, petri, z)
                    if not fichiers:
                        print(f"⚠ Aucun fichier pour {petri} {z} — ignoré")
                        continue
                    w, i = traiter_acquisitions_gellose(fichiers)
                    ip_irr.append(i)                 

        else:
            continue




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

spectres1 = []
etiquettes1 = []

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


                        spectres1.append(i)
                        etiquettes1.append(f"{souris}-{echantillon}-{zone}-{jour}-{dose}")

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

                    spectres1.append(i)
                    etiquettes1.append(f"{souris}-{zone}-{jour}-{dose}")               

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

                    spectres1.append(i)
                    etiquettes1.append(f"{souris}-{zone}-{jour}-{dose}")


# Cas spéciaux souris1.1 et souris2.1 (j8, petri3)
for souris_sp in ['souris1.1', 'souris2.1']:
    souris_label = souris_sp.replace('.', '_')
    for zone in ['zone1', 'zone2', 'zone3']:
        liste_fichiers = extraire_fichiers_jours_8_11('jour_8', 'petri3', souris_sp, zone)
        if liste_fichiers:
            w, i = traiter_acquisitions_verre(liste_fichiers)
            if i is not None and np.isfinite(i).all():
                spectres1.append(i)
                etiquettes1.append(f"{souris_label}-jour_8-45gy + P")


X1 = np.array(spectres1)

def parser_jour_dose(etiquettes):
    jours, doses = [], []
    for e in etiquettes:
        reste, jour, dose = e.rsplit('-', 2)
        jours.append(jour)
        doses.append(dose)
    return np.array(jours), np.array(doses)

jours_verre, doses_verre = parser_jour_dose(etiquettes1)

i_non_irr_arr = np.array(i_non_irr)
i_irr_arr = np.array(i_irr)
ip_irr_arr = np.array(ip_irr)
ip_non_irr_arr = np.array(ip_non_irr)

i_non_irr_moy = np.mean(i_non_irr_arr, axis=0)
i_irr_moy = np.mean(i_irr_arr, axis=0)
ip_irr_moy = np.mean(ip_irr_arr, axis=0)
ip_non_irr_moy = np.mean(ip_non_irr_arr, axis=0)

std_non_irr = np.std(i_non_irr_arr, axis=0)
std_irr = np.std(i_irr_arr, axis=0)
stdp_irr = np.std(ip_irr_arr, axis=0)
stdp_non_irr = np.std(ip_non_irr_arr, axis=0)

plt.plot(w, i_non_irr_moy, label='Non-irradiated', color='xkcd:royal blue', lw=0.8)
#up = i_non_irr_moy + std_non_irr
#low = i_non_irr_moy - std_non_irr
#plt.fill_between(w, low, up, color='xkcd:royal blue', alpha=0.2)

plt.plot(w, i_irr_moy, label='Irradiated NT', color='xkcd:scarlet', lw=0.8)
#up = i_irr_moy + std_irr
#low = i_irr_moy - std_irr
#plt.fill_between(w, low, up, color='xkcd:scarlet', alpha=0.2)

plt.plot(w, ip_irr_moy, label='Irradiated +P', color='xkcd:violet', lw=0.8)
#up = ip_irr_moy + stdp_irr
#low = ip_irr_moy - stdp_irr
#plt.fill_between(w, low, up, color='xkcd:violet', alpha=0.2)

plt.plot(w, ip_non_irr_moy, label='Non-irradiated +P', color='xkcd:green', lw=0.8)
#up = i_non_irr_moy + stdp_non_irr
#low = i_non_irr_moy - stdp_non_irr
#plt.fill_between(w, low, up, color='xkcd:scarlet', alpha=0.2)

plt.title('Medium spectrum of irradiated and non-irradiated skin')
plt.xlabel('Raman shift (cm⁻¹)')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()




                    
