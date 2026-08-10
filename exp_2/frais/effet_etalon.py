import matplotlib.pyplot as plt
import os
import glob
from extract_data import caracteriser_motif_fixe, corriger_motif_fixe, raman_shift_to_nm, traiter_acquisitions


from scipy.signal import savgol_filter
import numpy as np

def estimer_periode_frange(intensite_ref_brute, fenetre_grossiere=201):
    # Isoler les oscillations
    tendance_grossiere = savgol_filter(intensite_ref_brute, fenetre_grossiere, 3)
    residu = intensite_ref_brute - tendance_grossiere
    
    # FFT pour trouver la fréquence dominante
    spectre_freq = np.abs(np.fft.rfft(residu))
    freqs = np.fft.rfftfreq(len(residu))
    freq_dominante = freqs[np.argmax(spectre_freq[1:]) + 1]  # ignore la composante DC
    
    periode_pixels = 1 / freq_dominante if freq_dominante > 0 else None
    return periode_pixels



racine = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\exp_1\spectre_lumière_blanche"
fichiers = sorted(glob.glob(os.path.join(racine, '*.txt')))
wn_ref,_, intensite_ref_brute = traiter_acquisitions(fichiers)
t_lambda1, lisse1 = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, fenetre_lissage=101, ordre_poly=3, methode='savgol')
t_lambda2, lisse2 = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, fenetre_lissage=51, ordre_poly=3, methode='savgol')
t_lambda3, lisse3 = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute, fenetre_lissage=151, ordre_poly=3, methode='savgol')


print(estimer_periode_frange(intensite_ref_brute))

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

periode = 31.571428571428566

for facteur in [2, 3, 4, 5]:
    fenetre = int(round(facteur * periode))
    if fenetre % 2 == 0:
        fenetre += 1
    lisse = savgol_filter(intensite_ref_brute, fenetre, 4)
    residu = intensite_ref_brute - lisse
    
    plt.figure()
    plt.plot(residu)
    plt.title(f"Facteur={facteur}, fenêtre={fenetre}")
    plt.show()


plt.plot(wn_ref, intensite_ref_brute, label='Lumière blanche brute')
plt.plot(wn_ref, lisse1, label='f=101, ordre=3')
#plt.plot(wn_ref, t_lambda2+1, label='Lambda x2000, f=51, ordre=3')
plt.plot(wn_ref, lisse3, label='Lambda x2000, f=151, ordre=3')
plt.ylabel('Intensité')
plt.xlabel('Raman shift (cm⁻¹)')
plt.legend()
plt.tight_layout()
plt.show()