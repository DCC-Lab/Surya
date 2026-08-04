import matplotlib.pyplot as plt
import os
import glob
from extract_data import caracteriser_motif_fixe, corriger_motif_fixe, raman_shift_to_nm, traiter_acquisitions





racine = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\exp_1\spectre_lumière_blanche"
fichiers = sorted(glob.glob(os.path.join(racine, '*.txt')))
wn_ref,_, intensite_ref_brute = traiter_acquisitions(fichiers)
t_lambda1, lisse1 = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, fenetre_lissage=101, ordre_poly=3, methode='savgol')
t_lambda2, lisse2 = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, fenetre_lissage=100, ordre_poly=3, methode='savgol')
t_lambda3, lisse3 = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, fenetre_lissage=101, ordre_poly=4, methode='savgol')

plt.plot(wn_ref, intensite_ref_brute, label='Lumière blanche brute')
plt.plot(wn_ref, lisse1, label='Lambda x2000, f=101, ordre=3')
plt.plot(wn_ref, lisse2, label='Lambda x2000, f=100, ordre=3')
plt.plot(wn_ref, lisse3, label='Lambda x2000, f=101, ordre=4')
plt.ylabel('Intensité')
plt.xlabel('Raman shift (cm⁻¹)')
plt.legend()
plt.tight_layout()
plt.show()