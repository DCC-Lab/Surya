import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import glob
from extract_zone import formater_donnees, retirer_rayons_cosmiques, extraire_fichiers_jours_8_11, extraire_fichiers_jour_4, extraire_fichiers_j2_fixe,  soustraire_spectre, corriger_fluorescence

import os
 
# ─────────────────────────────────────────────────────────────
# CORRECTION DE L'EFFET D'ÉTALON (motif fixe multiplicatif du CCD)
# ─────────────────────────────────────────────────────────────
#
# Contrairement à la fluorescence (fond additif, lentement variable),
# l'effet d'étalon module le GAIN du détecteur de façon multiplicative
# et périodique en LONGUEUR D'ONDE (nm), pas en Raman shift (cm-1).
# Il faut donc:
#   1) le caractériser sur un spectre de référence lisse (avant conversion
#      en Raman shift, idéalement dès l'acquisition en nm)
#   2) diviser tous les spectres bruts par ce motif AVANT toute soustraction
#      de fond (bubblefill, soustraction de verre/gellose, etc.)


 
def caracteriser_motif_fixe(wn_ref, intensite_ref_brute,
                             fenetre_lissage=101, ordre_poly=3,
                             methode='savgol'):
    """
    Caractérise la fonction de motif fixe t(λ) à partir d'un spectre de
    référence spectralement lisse (ex: verre fluorescent NIST SRM 2241,
    ou toute source dont la vraie émission ne devrait pas osciller).
 
    wn_ref              : axe (longueur d'onde ou nombre d'onde, doit être
                           le même axe utilisé pour vos acquisitions brutes,
                           idéalement en nm avant conversion Raman)
    intensite_ref_brute : spectre brut mesuré de la référence
    fenetre_lissage     : taille de fenêtre Savitzky-Golay (impair, large
                           pour ne PAS suivre les oscillations d'étalon,
                           typiquement > 2x la période des franges)
    ordre_poly          : ordre du polynôme local pour Savitzky-Golay
    methode             : 'savgol' ou 'spline'
 
    Retourne t(λ), la fonction de motif fixe (sans dimension, ~1 en moyenne).
    """
    if methode == 'savgol':
        if fenetre_lissage % 2 == 0:
            fenetre_lissage += 1
        lisse = savgol_filter(intensite_ref_brute, fenetre_lissage, ordre_poly)
    elif methode == 'spline':
        spl = UnivariateSpline(wn_ref, intensite_ref_brute, s=len(wn_ref) * 50)
        lisse = spl(wn_ref)
    else:
        raise ValueError("methode doit être 'savgol' ou 'spline'")
 
    # Évite division par ~0
    lisse = np.where(np.abs(lisse) < 1e-9, 1e-9, lisse)
 
    t_lambda = intensite_ref_brute / lisse
    return t_lambda, lisse
 
 
def corriger_motif_fixe(wn_echantillon, intensite_echantillon,
                         wn_ref, t_lambda):
    """
    Applique la correction de motif fixe à un spectre échantillon.
    Interpole t_lambda sur la grille de l'échantillon si nécessaire.
    """
    if len(wn_ref) != len(wn_echantillon) or not np.allclose(wn_ref, wn_echantillon):
        t_interp = np.interp(wn_echantillon, wn_ref, t_lambda)
    else:
        t_interp = t_lambda
 
    return intensite_echantillon / t_interp

def traiter_acquisitions(liste_fichiers,
                          retirer_cosmiques=True, retirer_etalon=True, retirer_fluorescence=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Retourne (wavenumbers, spectre_somme).
    """
    spectres = []
    wn_ref = None

    if not liste_fichiers:  # ← vérifie si la liste est vide
        #print("Aucun fichier à traiter!")
        return None, None

    for fichier in liste_fichiers:
        wn, intensite = formater_donnees(fichier)

        if wn is None:  # ← saute les fichiers mal formatés
            continue

        if wn_ref is None:
            wn_ref = wn
        
        # retrait des rayons cosmiques
        if retirer_cosmiques:
            intensite = retirer_rayons_cosmiques(intensite)

        # Interpoler sur la grille de référence si longueur différente
        if len(wn) != len(wn_ref):
            intensite = np.interp(wn_ref, wn, intensite)

        # ajout à la liste des spectres
        spectres.append(intensite)
    
    # Moyennage des acquisitions : on a maintenant 1 spectre pour les 20 ou 30 acquisitions
    intensite_avec_fluo = np.mean(spectres, axis=0)



    # retrait de la fluorescence
    if retirer_fluorescence:
        intensite_sans_fluorescence = corriger_fluorescence(intensite_avec_fluo, min_bubble_widths=50, fit_order=1)

    return wn_ref, intensite_sans_fluorescence, intensite_avec_fluo, spectres


def raman_shift_to_nm(shift_cm1, laser_nm):
    nu_laser = 1e7 / laser_nm          # cm^-1
    nu_scattered = nu_laser - shift_cm1  # Stokes
    return 1e7 / nu_scattered           # nm


import matplotlib.pyplot as plt


racine = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\spectre_lumière_blanche"
fichiers = sorted(glob.glob(os.path.join(racine, '*.txt')))






wn_ref,_, intensite_ref_brute, spectres = traiter_acquisitions(fichiers)
t_lambda, lisse = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute)
i_corr_F= corriger_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, raman_shift_to_nm(wn_ref, 785), t_lambda)


#sans etalon ni sans rayonnement cosmique
w_j2s4p4, _, i_j2s4p4, s = traiter_acquisitions(extraire_fichiers_j2_fixe('verre', 'jour_2', 'petri4', 'souris4', 'zone1')) #petri4 = 60gy
i_j2s4p4_corr = corriger_motif_fixe(raman_shift_to_nm(w_j2s4p4, 785), i_j2s4p4, raman_shift_to_nm(wn_ref, 785), t_lambda)
w_j2s5p4, _, i_j2s5p4, s = traiter_acquisitions(extraire_fichiers_j2_fixe('verre', 'jour_2', 'petri4', 'souris5', 'zone1'))
i_j2s5p4_corr = corriger_motif_fixe(raman_shift_to_nm(w_j2s5p4, 785), i_j2s5p4, raman_shift_to_nm(wn_ref, 785), t_lambda)

#sans fluorescence, sans etalon ni rayon cosmique
i_j2s4p4_corr_SF = corriger_fluorescence(i_j2s4p4_corr)
i_j2s5p4_corr_SF = corriger_fluorescence(i_j2s5p4_corr)



#sans verre, sans fluorescence, sans etalon ni rayon cosmique
dossier_verre = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier_verre, "*.txt")))
wn_verre,_, i_verre, _ = traiter_acquisitions(liste_fichiers_verre)
i_verre_corr = corriger_motif_fixe(raman_shift_to_nm(wn_verre, 785), i_verre, raman_shift_to_nm(wn_ref, 785), t_lambda)
i_verre_corr_SF = corriger_fluorescence(i_verre_corr)

i_j2s4p4_corr_SF_SV = soustraire_spectre(w_j2s4p4, i_j2s4p4_corr_SF, wn_verre, i_verre_corr_SF)
i_j2s5p4_corr_SF_SV = soustraire_spectre(w_j2s5p4, i_j2s5p4_corr_SF, wn_verre, i_verre_corr_SF)


#sans verre, sans fluorescence ni rayon cosmique
i_verre_SF = corriger_fluorescence(i_verre)
i_j2s4p4_SF = corriger_fluorescence(i_j2s4p4)
i_j2s5p4_SF = corriger_fluorescence(i_j2s5p4)
i_j2s4p4_SF_SV = soustraire_spectre(w_j2s4p4, i_j2s4p4_SF, wn_verre, i_verre_SF)
i_j2s5p4_SF_SV = soustraire_spectre(w_j2s5p4, i_j2s5p4_SF, wn_verre, i_verre_SF)

# AVANT bubblefill : les deux devraient différer visiblement
plt.figure()
plt.plot(w_j2s4p4, i_j2s4p4_corr, label='avec correction étalon')
plt.plot(w_j2s4p4, i_j2s4p4, label='sans correction étalon')
plt.legend(); plt.title("Avant BubbleFill")

# APRÈS bubblefill (ce que fait ton pipeline final) : comparer l'écart-type résiduel
diff_apres = i_j2s4p4_corr_SF - i_j2s4p4_SF
print("std diff après fluorescence :", np.std(diff_apres))


fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # ← syntaxe correcte

ax1 = axes[0]   # ← pas des listes, des vrais axes
ax2 = axes[1]


ax1.plot(raman_shift_to_nm(w_j2s4p4, 785), i_j2s4p4_corr_SF_SV, 'b-', label='corr')
ax1.plot(raman_shift_to_nm(w_j2s4p4, 785), i_j2s4p4_SF_SV+200, 'r-', label='ref')
ax1.set_title('Spectre raman souris 4 irradiée 60 Gy')
ax2.plot(raman_shift_to_nm(w_j2s5p4, 785), i_j2s5p4_corr_SF_SV, 'b-', label='corr')
ax2.plot(raman_shift_to_nm(w_j2s5p4, 785), i_j2s5p4_SF_SV+200, 'r-', label='ref')
ax2.set_title('Spectre raman souris 5 irradiée 60 Gy')
plt.xlabel('Longueur d\'onde (nm)')
plt.ylabel('Intensité')
plt.legend()
plt.show()




#fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # ← syntaxe correcte

#ax1 = axes[0]   # ← pas des listes, des vrais axes
#ax2 = axes[1]


#ax1.plot(w_j2s4p4, i_j2s4p4, 'b-', label='référence')
#ax1.plot(w_j2s4p4, i_j2s4p4_corr+80, 'r-', label='corrigé')
#ax1.set_title('Spectre raman souris 4 irradiée 60 Gy')
#ax2.plot(w_j2s5p4, i_j2s5p4, 'b-', label='référence')
#ax2.plot(w_j2s5p4, i_j2s5p4_corr+80, 'r-', label='corrigé')
#ax2.set_title('Spectre raman souris 5 irradiée 60 Gy')
#plt.xlabel('Longueur d\'onde (nm)')
#plt.ylabel('Intensité')
#plt.legend()
#plt.show()
