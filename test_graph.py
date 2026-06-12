from extract_data import lecteur_fichier_j4, traiter_acquisitions_et_verre
import matplotlib.pyplot as plt

w_j4s1p4, i_j4s1p4 = traiter_acquisitions_et_verre(lecteur_fichier_j4('jour4', 'petri1', 'souris4'))

plt.figure(figsize=(10, 6))
plt.plot(w_j4s1p4, i_j4s1p4, label="on moyenne avant bubblefill avant d'enlever le spectre du verre")
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Intensity')
plt.title('Spectre de la souris 1 --- jour 4  --- 45gy + P')
plt.legend()
plt.show()
