from LDA import charger_nocif, charger_spectres, parser_etiquettes, matrice_confusion, etiquettes, afficher_ld1
from extract_data import adjust_spectrum, extract_frais, extract_fixe, lecteur_données_moy_fixe, lecteur_données_moy_frais

import numpy as np
import matplotlib.pyplot as plt


lecteurs = {
    'frais':extract_frais,
    'fixe':extract_fixe,
    'moyenfrais':lecteur_données_moy_frais,
    'moyenfixe': lecteur_données_moy_fixe
}

w1, i1 = adjust_spectrum(lecteurs['frais']('batch#1', 'petri1', 'z1'), retirer_nocif=True)
w2, i2 = adjust_spectrum(lecteurs['frais']('batch#1', 'petri1', 'z1'), retirer_nocif=False)

plt.plot(w1, i1, label='corrigé')
plt.plot(w2, i2, label='non corrigé')
plt.legend()
plt.show()




echantillons, doses, sexes, traitements, souris_id, etats, zones = parser_etiquettes(etiquettes)


masque1 = (etats == 'frais') & (traitements == 'NT') & (sexes == 'F')
matrice_confusion(masque1, 'dose', 'Effet de la dose, femelles non traités')

masque2 = (etats == 'frais') & (doses == 45) & (sexes == 'F')
matrice_confusion(masque2, 'traitement', 'Effet du traitement, femelles irradiées')

matrice_confusion(masque1, 'dose', "Effet dose, femelles non traitées", graph=True, couleur='xkcd:scarlet')
matrice_confusion(masque2, 'traitement', "Effet pansement, femelles irradiées", graph=True, couleur='tab:green')

afficher_ld1()   # affiche les deux courbes superposées sur le même graphique