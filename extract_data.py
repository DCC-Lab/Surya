import numpy as np
import os
from orpl.baseline_removal import bubblefill
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np

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
# 3. CORRECTION DU SPECTRE DU VERRE
# ─────────────────────────────────────────────


def soustraire_verre(wn_echantillon, intensite_echantillon, 
                     wn_verre, intensite_verre):
    """
    Soustrait la contribution du verre en trouvant le meilleur coefficient.
    Utilise NNLS pour que le coefficient soit toujours positif.
    """
    # Interpoler le verre sur la même grille que l'échantillon
    verre_interp = np.interp(wn_echantillon, wn_verre, intensite_verre)
    
    # Trouver le coefficient α optimal (NNLS = non-negative least squares)
    A = verre_interp.reshape(-1, 1)
    alpha, _ = nnls(A, intensite_echantillon)
    
    # Soustraire
    intensite_corrigee = intensite_echantillon - alpha * verre_interp
    
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
# 6. RETRAITS DU VERRE + CENTRAGE DES DONNÉES
# ────────────────────────────────────────────────────────────────────────


dossier = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier, "*.txt")))


def traiter_acquisitions_et_verre(liste_fichiers, wn_min=500, wn_max=3025, retirer_cosmiques=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre du verre et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    
    wn, i = traiter_acquisitions(liste_fichiers, wn_min, wn_max, retirer_cosmiques)
    wn_verre, i_verre = traiter_acquisitions(liste_fichiers_verre, wn_min, wn_max, retirer_cosmiques)
    intensite_SV = soustraire_verre(wn, i, wn_verre, i_verre)
    intensité_SV_SF = corriger_fluorescence(intensite_SV, min_bubble_widths=50, fit_order=1)
    return wn, intensité_SV_SF - np.mean(intensité_SV_SF)



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

# ─────────────────────────────────────────────
# CONSTRUCTION DE LA MATRICE DE DONNÉES
# ─────────────────────────────────────────────
# DIFFÉRENCIER PETRI DE ZONE !!!!!!!!!!!!!!
"""
Construit une matrice de données pour les jours 2,
4, 8 et 11. va avoir la forme :

                    500 nm^-1  ...     3000 nm^-1 
souris1-j2-0gy      0,3       ...       0,5
souris1-j2-45gy     0,2       ...       0,4
        ...
souris5-j11-80gy    0,1       ...       0,3
""" 
spectres = []
etiquettes = []

dose_j2_j8 = ['0gy', '45gy', '45gy + P', '60gy', '80gy']
dose_j4 = ['60gy', '80gy', '0gy', '45gy + P', '45gy' ]
dose_j11 = ['0gy', '45gy', '60gy', '80gy']
liste_souris = ['souris1', 'souris2', 'souris3', 'souris4', 'souris5']
liste_petri = ['petri1', 'petri2', 'petri3', 'petri4', 'petri5']
liste_jour = ['jour2', 'jour4', 'jour_8', 'jour_11']

for souris in liste_souris:
    for petri in liste_petri:
        idx_petri = liste_petri.index(petri)  # ← calcule une fois, réutilise partout
        
        for jour in liste_jour:
            if jour == 'jour2':
                liste_fichiers = lecteur_fichier_j2(jour, petri, souris)
                if not liste_fichiers:
                    #print(f"Aucun fichier : {souris}, {petri}, {jour}")
                    continue
                w, i = traiter_acquisitions_et_verre(liste_fichiers)
                dose = dose_j2_j8[idx_petri]

            elif jour == 'jour4':
                liste_fichiers = lecteur_fichier_j4(jour, petri, souris)
                if not liste_fichiers:
                    #print(f"Aucun fichier : {souris}, {petri}, {jour}")
                    continue
                w, i = traiter_acquisitions_et_verre(liste_fichiers)  # ← liste_fichiers manquait
                dose = dose_j4[idx_petri]

            elif jour == 'jour_8':
                # SOURIS 1.1 ET 2.1!!
                liste_fichiers = lecteur_fichier_j8_j11(jour, petri, souris)
                if not liste_fichiers:
                    #print(f"Aucun fichier : {souris}, {petri}, {jour}")
                    continue
                w, i = traiter_acquisitions_et_verre(liste_fichiers)
                dose = dose_j2_j8[idx_petri]

            elif jour == 'jour_11':
                if idx_petri >= len(dose_j11):   # ← protège contre petri5 qui n'existe pas en j11
                    continue
                liste_fichiers = lecteur_fichier_j8_j11(jour, petri, souris)
                if not liste_fichiers:
                    #print(f"Aucun fichier : {souris}, {petri}, {jour}")
                    continue
                w, i = traiter_acquisitions_et_verre(liste_fichiers)
                dose = dose_j11[idx_petri]

            # vérification NaN/Inf — même logique pour tous les jours
            if w is None or i is None:
                continue
            if not np.isfinite(i).all():
                print(f"NaN/Inf détectés : {souris}, {petri}, {jour} — spectre ignoré")
                continue

            spectres.append(i)
            etiquettes.append(f"{souris}-{jour}-{dose}")  # ← propre et cohérent

w_j8_p3s1_1, i_j8p3s1_1 = traiter_acquisitions_et_verre(lecteur_fichier_j8_j11('jour_8', 'petri3', 'souris1.1'))
spectres.append(i_j8p3s1_1)
etiquettes.append('souris1_1-j8-45gy + P')
w_j8_p3s1_2, i_j8p3s1_2 = traiter_acquisitions_et_verre(lecteur_fichier_j8_j11('jour_8', 'petri3', 'souris2.1'))
spectres.append(i_j8p3s1_2)
etiquettes.append('souris2_1-j8-45gy + P')

X = np.array(spectres)
print(f"Matrice X construite : {X.shape}")
quantité = 0
for i in etiquettes:
    if 'souris3' in i:
        print(i)
        quantité += 1
print(quantité)

