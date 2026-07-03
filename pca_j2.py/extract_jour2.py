import numpy as np
import os
from orpl.baseline_removal import bubblefill
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
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
                #print(f"temps d'intégration : {integration} pour {chemin_fichier}")
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

# 3. SOUSTRACTION DE SPECTRE NOCIFS
# ─────────────────────────────────────────────


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

def traiter_acquisitions(liste_fichiers,
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
    spectre_moyen = np.mean(spectres, axis=0)

    # retrait de la fluorescence
    if retirer_fluorescence:
        intensite_sans_fluorescence = corriger_fluorescence(spectre_moyen, min_bubble_widths=50, fit_order=1)

    return wn_ref, intensite_sans_fluorescence

dossier_verre = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier_verre, "*.txt")))

dossier_gellose = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\spectre_gellose"
liste_fichiers_gellose = sorted(glob.glob(os.path.join(dossier_gellose, "*.txt")))

def traiter_acquisitions_verre(liste_fichiers, retirer_cosmiques=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre du verre et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    
    wn, i = traiter_acquisitions(liste_fichiers, retirer_cosmiques)
    wn_verre, i_verre = traiter_acquisitions(liste_fichiers_verre, retirer_cosmiques)
    intensite_SV = soustraire_spectre(wn, i, wn_verre, i_verre)
    intensité_SV_SF = corriger_fluorescence(intensite_SV, min_bubble_widths=50, fit_order=1)
    
    intensite_centree = intensité_SV_SF - np.mean(intensité_SV_SF)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml

def traiter_acquisitions_verre_gelose(liste_fichiers, retirer_cosmiques=True):

    wn, i = traiter_acquisitions(liste_fichiers, retirer_cosmiques)
    wn_verre, i_verre = traiter_acquisitions(liste_fichiers_verre, retirer_cosmiques)
    wn_gelose, i_gelose = traiter_acquisitions(liste_fichiers_gellose, retirer_cosmiques)
    intensite_SV = soustraire_spectre(wn, i, wn_verre, i_verre)
    intensite_SV_SG = soustraire_spectre(wn, intensite_SV, wn_gelose, i_gelose)
    intensite_SV_SG_SF = corriger_fluorescence(intensite_SV_SG, min_bubble_widths=50, fit_order=1)

    intensite_centree = intensite_SV_SG_SF - np.mean(intensite_SV_SG_SF)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml

# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DE LA GELLOSE + CENTRAGE DES DONNÉES: JOUR 2 ET 4
# ────────────────────────────────────────────────────────────────────────



def traiter_acquisitions_gellose(liste_fichiers, retirer_cosmiques=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre de la gellose et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    wn, i = traiter_acquisitions(liste_fichiers, retirer_cosmiques)
    wn_gellose, i_gellose = traiter_acquisitions(liste_fichiers_gellose, retirer_cosmiques)
    # ── Vérification avant soustraction ──────────────────────────────────────
    if wn is None or i is None:
        print("❌ Échantillon : None")
        return None, None
    if wn_gellose is None or i_gellose is None:
        print("❌ Gellose : None")
        return None, None
    if not np.isfinite(i).all():
        print(f"❌ NaN/Inf dans l'échantillon : {np.sum(~np.isfinite(i))} points")
        return None, None
    if not np.isfinite(i_gellose).all():
        print(f"❌ NaN/Inf dans la gellose : {np.sum(~np.isfinite(i_gellose))} points")
        return None, None
    # ─────────────────────────────────────────────────────────────────────────
    intensite_SG = soustraire_spectre(wn, i, wn_gellose, i_gellose)
    intensité_SG_SF = corriger_fluorescence(intensite_SG, min_bubble_widths=50, fit_order=1)
    
    intensite_centree = intensité_SG_SF - np.mean(intensité_SG_SF)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml

# ───────────────────────────────────────────────────
# on va créer des fonctions 
# pour extraire les fichiers par zone et non par 
# souris
#────────────────────────────────────────────────────


#───────────────────────────────────────────2. FICHIER DU JOUR 2 ────────────────────────────────────────────

racine2 = r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya"

def extraire_fichiers_j2_frais(jour, petri, souris, zone):
    dossier = os.path.join(racine2, jour, "raman", petri)

    if petri == 'petri1' and souris == 'souris1':
        pattern = os.path.join(dossier, f"{souris}*")

    else:
        pattern = os.path.join(dossier, f"{souris}*{zone}*")

    dossiers_trouves = sorted(glob.glob(pattern))
    
    if not dossiers_trouves:
        #print(f"Pour le {jour} La {zone} de la {souris} du {petri} n'existe pas")
        return []

    fichiers_zone = sorted(glob.glob(os.path.join(dossiers_trouves[0], "*.txt")))
    #print(f'Fichiers de la {zone} du {petri} du {jour} : {fichiers_zone}')
    
    return fichiers_zone


racine1 = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya"

def extraire_fichiers_j2_fixe(matiere, jour, petri, souris, zone, fichiers_par_zone=10):

    dossier = os.path.join(racine1, jour, "raman", petri)
    pattern = os.path.join(dossier, f"{souris}*{matiere}*")

    tous_les_fichiers = sorted(glob.glob(pattern))
    
    if not tous_les_fichiers:
        #print(f"Aucun fichier trouvé avec le pattern : {pattern}")
        return []
    
    # Trie par date de modification (le plus ancien en premier)
    tous_les_fichiers_tries = sorted(tous_les_fichiers, key=lambda f: os.path.getmtime(f))
    
    # Découpe en tranches de 10
    indice = int(zone[-1])
    debut = (indice - 1) * fichiers_par_zone
    fin = debut + fichiers_par_zone
    fichiers_zone = tous_les_fichiers_tries[debut:fin]
    #print(f"{zone} — {len(fichiers_zone)} fichiers trouvés: {fichiers_zone}")
    
    return fichiers_zone



    

  