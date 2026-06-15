import numpy as np
import os
from orpl.baseline_removal import bubblefill
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. LECTURE ET TRONCATURE
# ─────────────────────────────────────────────

def formater_donnees(chemin_fichier, wn_min=500, wn_max=3025):
    data = []
    integration = 1.0  # valeur par défaut si non trouvée
    
    with open(chemin_fichier, 'r') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith('#') or ligne.startswith('>'):
                continue
            if 'Integration Time' in ligne:
                valeur_str = ligne.split(':')[-1].strip().replace(',', '.')
                integration = float(valeur_str)
                print(f"temps d'intégration : {integration}")
                continue
            try:
                valeurs = [float(x) for x in ligne.replace(',', '.').split()]
                if len(valeurs) >= 2:
                    data.append(valeurs[:2])
            except ValueError:
                continue

    if len(data) == 0:
        #print(f"Fichier vide ou mal formaté : {chemin_fichier}")
        return None, None

    data = np.array(data)
    
    if data.ndim != 2:
        #print(f"Format inattendu : {chemin_fichier}")
        return None, None

    wn = data[:, 0]
    intensite = data[:, 1] / integration

    masque = (wn >= wn_min) & (wn <= wn_max)
    return wn[masque], intensite[masque]


# ─────────────────────────────────────────────
# 2. RETRAIT DES RAYONS COSMIQUES
# ─────────────────────────────────────────────

def retirer_rayons_cosmiques(intensite, seuil=10.0, fenetre=5):
    """
    Détecte et remplace les spikes de rayons cosmiques.
    Méthode : un point est cosmique si son écart à la médiane locale
    dépasse (seuil × MAD locale).
    """
    intensite_corr = intensite.copy()
    n = len(intensite)
    demi = fenetre // 2

    for i in range(demi, n - demi):
        voisins = np.concatenate([intensite[i-demi:i], intensite[i+1:i+demi+1]])
        mediane = np.median(voisins)
        mad = np.median(np.abs(voisins - mediane)) + 1e-10  # évite division par zéro
        if abs(intensite[i] - mediane) > seuil * mad:
            # Remplace par interpolation linéaire des voisins
            intensite_corr[i] = np.interp(i,
                                           [i - demi, i + demi],
                                           [intensite[i - demi], intensite[i + demi]])
    return intensite_corr

# ─────────────────────────────────────────────
# 3. SOUSTRACTION DE SPECTRE NOCIFS
# ─────────────────────────────────────────────


def soustraire_spectre(wn_echantillon, intensite_echantillon, 
                     wn_nocif, intensite_nocif):
    """
    Soustrait la contribution du verre en trouvant le meilleur coefficient.
    Utilise NNLS pour que le coefficient soit toujours positif.
    """
    # Interpoler le verre sur la même grille que l'échantillon
    nocif_interp = np.interp(wn_echantillon, wn_nocif, intensite_nocif)
    
    # Trouver le coefficient α optimal (NNLS = non-negative least squares)
    A = nocif_interp.reshape(-1, 1)
    alpha, _ = nnls(A, intensite_echantillon)
    
    # Soustraire
    intensite_corrigee = intensite_echantillon - alpha * nocif_interp
    
    return intensite_corrigee

# ─────────────────────────────────────────────
# 4. CORRECTION DE FLUORESCENCE (baseline)
# ─────────────────────────────────────────────

def corriger_fluorescence(intensite, min_bubble_widths=50, fit_order=1):
    """
    Supprime l'autofluorescence avec l'algorithme BubbleFill (ORPL).
    
    wn                : tableau des nombres d'onde (cm⁻¹)
    intensite         : tableau des intensités brutes
    min_bubble_widths : largeur minimale des bulles en pixels (défaut: 50)
                        doit être > largeur du pic Raman le plus large
    fit_order         : ordre du polynôme de correction résiduelle (défaut: 1)
    
    Retourne (intensite_corrigee, baseline)
    """
    résultat = bubblefill(intensite, 
                           min_bubble_widths=min_bubble_widths, 
                           fit_order=fit_order)
    spectre_corrigé =  résultat[0]
    
    return spectre_corrigé


# ───────────────────────────────────────────────────────────
# 5. COMBINAISON DES ACQUISITIONS + RETRAITS RAYONS COSMIQUES
# ───────────────────────────────────────────────────────────


def traiter_acquisitions(liste_fichiers, wn_min=500, wn_max=3025,
                          retirer_cosmiques=True, retirer_fluorescence=True):
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
        wn, intensite = formater_donnees(fichier, wn_min, wn_max)

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
    spectre_moyen = np.mean(spectres, axis=0)

    # retrait de la fluorescence
    if retirer_fluorescence:
        intensite_sans_fluorescence = corriger_fluorescence(spectre_moyen, min_bubble_widths=50, fit_order=1)

    return wn_ref, intensite_sans_fluorescence





# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DU VERRE + CENTRAGE DES DONNÉES: JOUR 8 ET 11
# ────────────────────────────────────────────────────────────────────────


dossier = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier, "*.txt")))


def traiter_acquisitions_j8_j11(liste_fichiers, wn_min=500, wn_max=3025, retirer_cosmiques=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre du verre et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    
    wn, i = traiter_acquisitions(liste_fichiers, wn_min, wn_max, retirer_cosmiques)
    wn_verre, i_verre = traiter_acquisitions(liste_fichiers_verre, wn_min, wn_max, retirer_cosmiques)
    intensite_SV = soustraire_spectre(wn, i, wn_verre, i_verre)
    intensité_SV_SF = corriger_fluorescence(intensite_SV, min_bubble_widths=50, fit_order=1)
    return wn, intensité_SV_SF - np.mean(intensité_SV_SF)

# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DE LA GELLOSE + CENTRAGE DES DONNÉES: JOUR 2 ET 4
# ────────────────────────────────────────────────────────────────────────

dossier = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\spectre_gellose"
liste_fichier_verre = sorted(glob.glob(os.path.join(dossier, "*.txt")))

def traiter_acquisitions_j2_j4(liste_fichiers, wn_min=500, wn_max=3025, retirer_cosmiques=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre de la gellose et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    wn, i = traiter_acquisitions(liste_fichiers, wn_min, wn_max, retirer_cosmiques)
    wn_gellose, i_gellose = traiter_acquisitions(liste_fichiers_verre, wn_min, wn_max, retirer_cosmiques)
    intensite_SG = soustraire_spectre(wn, i, wn_gellose, i_gellose)
    intensité_SG_SF = corriger_fluorescence(intensite_SG, min_bubble_widths=50, fit_order=1)
    return wn, intensité_SG_SF - np.mean(intensité_SG_SF)

# ─────────────────────────────────────────────
# OBTENTEUR DE FICHIERS J2, J4, J8, J11
# ─────────────────────────────────────────────

#J8, J11

racine1 = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya"

def lecteur_fichier_j8_j11(jour, petri, souris):
    
    fichiers = []

    for i in range(1, 4):
        zone = f"zone{i}"
        dossier = os.path.join(racine1, jour, "Raman", petri, souris, zone)
        
        # Si le dossier n'existe pas, on le saute sans buguer
        if not os.path.exists(dossier):
            #print(f"Dossier absent, ignoré : {dossier}")
            continue
        
        # Chercher les fichiers .txt dans ce dossier
        fichiers_zone = glob.glob(os.path.join(dossier, "*.txt"))
        fichiers.extend(fichiers_zone)

    fichiers = sorted(fichiers)
    return fichiers

#J2

racine2 = r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya"

def lecteur_fichier_j2(jour, petri, souris):
    """
    Gère la structure : racine/jour/raman/petri/souris_dose_zone*/
    ex: souris1_0Gy/  ou  souris2_0Gy_zone1/  souris2_0Gy_zone2/
    """
    fichiers = []
    dossier_petri = os.path.join(racine2, jour, "raman", petri)

    # Cherche tous les dossiers qui commencent par le nom de la souris
    pattern = os.path.join(dossier_petri, f"{souris}*")
    dossiers_trouves = sorted(glob.glob(pattern))

    if not dossiers_trouves:
        return []

    for dossier in dossiers_trouves:
        if not os.path.isdir(dossier):
            continue
        fichiers_zone = sorted(glob.glob(os.path.join(dossier, "*.txt")))
        fichiers.extend(fichiers_zone)

    return fichiers

#J4

def lecteur_fichier_j4(jour, petri, souris):
    ''' 
    Gère la structure : racine/jour/raman/petri/souris_dose_zone*/
    cependant, les souris sont mélangées en un dossier
    '''
    dossier_petri = os.path.join(racine2, jour, "raman", petri)

    if not os.path.exists(dossier_petri):
        #print(f"Dossier absent : {dossier_petri}")
        return []   # ← retourne liste vide mais l'appelant continue

    # cherche tous les fichiers qui commencent par le nom de la souris
    pattern = os.path.join(dossier_petri, f"{souris}*.txt")
    fichiers = sorted(glob.glob(pattern))

    return fichiers


w_j2s4p4, i_j2s4p4 = traiter_acquisitions_j2_j4(lecteur_fichier_j2('jour2', 'petri4', 'souris4')) #petri4 = 60gy
w_j2s5p4, i_j2s5p4 = traiter_acquisitions_j2_j4(lecteur_fichier_j2('jour2', 'petri4', 'souris5'))

w_j4s4p1, i_j4s4p1 = traiter_acquisitions_j2_j4(lecteur_fichier_j4('jour4', 'petri1', 'souris4')) #petri1 = 60gy
w_j4s5p1, i_j4s5p1 = traiter_acquisitions_j2_j4(lecteur_fichier_j4('jour4', 'petri1', 'souris5'))

w_j8s4p4, i_j8s4p4 = traiter_acquisitions_j8_j11(lecteur_fichier_j8_j11('jour8', 'petri4', 'souris4')) #petri4 = 60gy
w_j8s5p4, i_j8s5p4 = traiter_acquisitions_j8_j11(lecteur_fichier_j8_j11('jour8', 'petri4', 'souris5'))

w_j11s4p3, i_j11s4p3 = traiter_acquisitions_j8_j11(lecteur_fichier_j8_j11('jour11', 'petri3', 'souris4')) #petri3 = 60gy
w_j11s5p3, i_j11s5p3 = traiter_acquisitions_j8_j11(lecteur_fichier_j8_j11('jour11', 'petri3', 'souris5'))






fig, axes = plt.subplots(1, 1, figsize=(10, 10))  # ← syntaxe correcte

ax1 = axes[0, 0]   # ← pas des listes, des vrais axes
ax2 = axes[1, 0]


ax1.plot(w_j2s4p4, i_j2s4p4, label='souris4')
ax1.plot(w_j4s4p1, i_j4s4p1, label='souris4')
ax1.plot(w_j8s4p4, i_j8s4p4, label='souris4')
ax1.plot(w_j11s4p3, i_j11s4p3, label='souris4')
ax1.set_title('Spectre raman souris 4 irradiée 60 Gy')

ax2.plot(w_j2s5p4, i_j2s5p4, label='souris5')
ax2.plot(w_j4s5p1, i_j4s5p1, label='souris5')
ax2.plot(w_j8s5p4, i_j8s5p4, label='souris5')
ax2.plot(w_j11s5p3, i_j11s5p3, label='souris5')
ax2.set_title('Spectre raman souris 5 irradiée 60 Gy')

plt.tight_layout()
plt.show()


