import os
import glob
from scipy.optimize import lsq_linear
import numpy as np
import matplotlib.pyplot as plt
from extract_data import traiter_acquisitions_gellose, lecteur_données_frais, lecteur_données_fixes, lecteur_données_moy, traiter_acquisitions_verre, extraire_fichiers_jour_0, extraire_fichiers_j2_fixe,  extraire_fichiers_jour_4, extraire_fichiers_jours_8_11


racine8 = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\exp_2"
def lecteur_données(batch, petri):
    dossier = os.path.join(racine8, batch, 'frais', petri)
    pattern = os.path.join(dossier, f"*petri*")
    tous_les_fichiers = sorted(glob.glob(pattern))
    
    if not tous_les_fichiers:
        print(f"Aucun fichier trouvé avec le pattern : {pattern}")
        return []

    return tous_les_fichiers




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


i_s = []


for batch, petri in config.items():
    for petri, (echantillon, dose, type_) in petri.items():
        fichiers = lecteur_données(batch, petri)
        if not fichiers:
            continue
        w, i = traiter_acquisitions_gellose(fichiers)
        i_s.append(i)



def soustraire_spectre(wn_echantillon, intensite_echantillon, 
                        wn_nocif, intensite_nocif,
                        ordre_baseline=1, fenetres_fit=None):
    """
    Soustrait la contribution du verre (ou de la gellose) en trouvant 
    le meilleur coefficient, avec une baseline polynomiale optionnelle 
    pour absorber le fond que le verre seul n'explique pas.

    ordre_baseline : ordre du polynôme de baseline ajouté au fit 
                      (0 = juste un offset, 1 = offset + pente, etc.)
                      Mets 0 pour te rapprocher de ton comportement original.
    fenetres_fit   : liste de tuples (wn_min, wn_max) pour restreindre 
                      le fit à des zones dominées par le verre 
                      (ex: [(500, 550), (900, 950)]). None = tout le spectre.
    """
    
    wn_echantillon = np.asarray(wn_echantillon, dtype=float)
    intensite_echantillon = np.asarray(intensite_echantillon, dtype=float)
    wn_nocif = np.asarray(wn_nocif, dtype=float)
    intensite_nocif = np.asarray(intensite_nocif, dtype=float)

    # ... reste du code inchangé
    # Interpoler le verre/gellose sur la même grille que l'échantillon
    nocif_interp = np.interp(wn_echantillon, wn_nocif, intensite_nocif)

    # Masque pour restreindre le fit à certaines fenêtres si demandé
    if fenetres_fit is not None:
        masque = np.zeros_like(wn_echantillon, dtype=bool)
        for (lo, hi) in fenetres_fit:
            masque |= (wn_echantillon >= lo) & (wn_echantillon <= hi)
    else:
        masque = np.ones_like(wn_echantillon, dtype=bool)

    # Matrice de design : [verre, 1, x, x², ...] (x normalisé pour la stabilité numérique)
    x_norm = (wn_echantillon - wn_echantillon.mean()) / wn_echantillon.std()
    colonnes = [nocif_interp] + [x_norm**k for k in range(ordre_baseline + 1)]
    A_full = np.column_stack(colonnes)

    A_fit = A_full[masque]
    y_fit = intensite_echantillon[masque]

    # Bornes : coefficient du verre >= 0, coefficients de baseline libres
    n_baseline = ordre_baseline + 1
    bornes_inf = [0.0] + [-np.inf] * n_baseline
    bornes_sup = [np.inf] + [np.inf] * n_baseline

    resultat = lsq_linear(A_fit, y_fit, bounds=(bornes_inf, bornes_sup))
    coeffs = resultat.x
    alpha = coeffs[0]

    # Appliquer le modèle complet (verre + baseline) sur TOUT le spectre
    modele_complet = A_full @ coeffs
    intensite_corrigee = intensite_echantillon - modele_complet

    return intensite_corrigee


fichiers = lecteur_données_frais('batch#1', 'petri2', 'z1')
if not fichiers:
    print(f"⚠ Aucun fichier pour {petri} — ignoré")

w, i_p1z1 = traiter_acquisitions_gellose(fichiers)





i_s_arr = np.array(i_s)


i_s_moy = np.mean(i_s_arr, axis=0)

std_s = np.std(i_s_arr, axis=0)

wg1, ip1z1 = traiter_acquisitions_gellose(lecteur_données('batch#1', 'petri1'))
i_corr1 = soustraire_spectre(w, i_p1z1, w, i_s_moy, ordre_baseline=1, fenetres_fit=None)
i_corr2 = soustraire_spectre(w, i_p1z1, w, i_s_moy, ordre_baseline=2, fenetres_fit=None)



plt.plot(w, i_s_moy, label='Spectre gélose moyenné', color='xkcd:royal blue', lw=0.8)
up = i_s_moy + std_s
low = i_s_moy - std_s
plt.fill_between(w, low, up, color='xkcd:royal blue', alpha=0.2)

plt.plot(w, i_corr1, label='corrigé')
plt.plot(w, i_p1z1, label='non corrigé')

plt.title('Petri 2 z1')
plt.xlabel('Raman shift (cm⁻¹)')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()

