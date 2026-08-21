from LDA import parser_etiquettes, matrice_confusion, etiquettes, afficher_ld1
from extract_data import adjust_spectrum, extract_frais, extract_fixe, lecteur_données_moy_fixe, lecteur_données_moy_frais

import numpy as np
import matplotlib.pyplot as plt


lecteurs = {
    'frais':extract_frais,
    'fixe':extract_fixe,
    'moyenfrais':lecteur_données_moy_frais,
    'moyenfixe': lecteur_données_moy_fixe
}

# w1, i1 = adjust_spectrum(lecteurs['frais']('batch#1', 'petri1', 'z1'), retirer_nocif=True)
# w2, i2 = adjust_spectrum(lecteurs['frais']('batch#1', 'petri1', 'z1'), retirer_nocif=False)

# plt.plot(w1, i1, label='corrigé')
# plt.plot(w2, i2, label='non corrigé')
# plt.legend()
# plt.show()




echantillons, doses, sexes, traitements, souris_id, etats, zones = parser_etiquettes(etiquettes)


masque1 = (etats == 'frais') & (traitements == 'NT') & (sexes == 'F')
info1 = matrice_confusion(masque1, 'dose', 'Effet de la dose, femelles non traités')

masque2 = (etats == 'frais') & (doses == 45) & (sexes == 'F')
info2 = matrice_confusion(masque2, 'traitement', 'Effet du traitement, femelles irradiées')



afficher_ld1(info1,'Effet traitement et effet dose', info2)