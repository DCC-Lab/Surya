import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import os
from orpl.baseline_removal import bubblefill
import glob
from scipy.optimize import lsq_linear
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


















# ─────────────────────────────────────────────
# 2. retrait de l'étalon
# ─────────────────────────────────────────────

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


def raman_shift_to_nm(shift_cm1, laser_nm):
    nu_laser = 1e7 / laser_nm          # cm^-1
    nu_scattered = nu_laser - shift_cm1  # Stokes
    return 1e7 / nu_scattered           # nm


























    

def retirer_rayons_cosmiques(wn, intensite, seuil=10.0, fenetre=5, zones_protegees=None):
    """
    Détecte et remplace les spikes de rayons cosmiques.
    
    zones_protegees : liste de tuples (wn_min, wn_max) à exclure du filtrage,
                       pour préserver de vrais pics Raman étroits et 
                       reproductibles (ex: [(2790, 2820)]).
    """
    intensite_corr = intensite.copy()
    n = len(intensite)
    demi = fenetre // 2

    if zones_protegees is not None:
        masque_protege = np.zeros(n, dtype=bool)
        for (lo, hi) in zones_protegees:
            masque_protege |= (wn >= lo) & (wn <= hi)
    else:
        masque_protege = np.zeros(n, dtype=bool)

    for i in range(demi, n - demi):
        if masque_protege[i]:
            continue  # on ne touche pas à cette zone
        voisins = np.concatenate([intensite[i-demi:i], intensite[i+1:i+demi+1]])
        mediane = np.median(voisins)
        mad = np.median(np.abs(voisins - mediane)) + 1e-10
        if abs(intensite[i] - mediane) > seuil * mad:
            intensite_corr[i] = np.interp(i, [i - demi, i + demi],
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


def corriger_fluorescence(intensite, min_bubble_widths=90, fit_order=1):
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

from scipy import sparse
from scipy.sparse.linalg import spsolve

def corriger_fluorescence_als(intensite, lam=1e7, p=0.01, n_iter=10):
    """
    Supprime l'autofluorescence avec l'algorithme ALS 
    (Asymmetric Least Squares, Eilers & Boersma 2005).
    """
    intensite = np.asarray(intensite, dtype=float)
    n = len(intensite)

    # Matrice de différences secondes, shape (n-2, n)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), dtype=float)
    # Dᵀ @ D donne une matrice (n, n) qui pénalise la courbure
    DtD = lam * (D.transpose().dot(D))

    poids = np.ones(n)
    W = sparse.spdiags(poids, 0, n, n)

    baseline = np.zeros(n)
    for _ in range(n_iter):
        W.setdiag(poids)
        Z = W + DtD
        baseline = spsolve(Z.tocsc(), poids * intensite)

        poids = p * (intensite > baseline) + (1 - p) * (intensite <= baseline)

    intensite_corrigee = intensite - baseline

    return intensite_corrigee





def traiter_acquisitions(liste_fichiers,
                          retirer_cosmiques=True, retirer_fluorescence=True, zones_protegees=[(1050, 1070),(2780, 2820)]):
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
            intensite = retirer_rayons_cosmiques(wn_ref, intensite, zones_protegees=zones_protegees)

        # Interpoler sur la grille de référence si longueur différente
        if len(wn) != len(wn_ref):
            intensite = np.interp(wn_ref, wn, intensite)

        # ajout à la liste des spectres
        spectres.append(intensite)
    
    # Moyennage des acquisitions : on a maintenant 1 spectre pour les 20 ou 30 acquisitions
    spectre_moyen = np.mean(spectres, axis=0)



    # retrait de la fluorescence
    if retirer_fluorescence:
        intensite_sans_fluorescence = corriger_fluorescence_als(spectre_moyen)

    return wn_ref, intensite_sans_fluorescence, spectre_moyen

dossier_verre = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\exp_1\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier_verre, "*.txt")))
wn_verre, i_verre, _ = traiter_acquisitions(liste_fichiers_verre)

dossier_gellose = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\exp_1\spectre_gellose"
liste_fichiers_gellose = sorted(glob.glob(os.path.join(dossier_gellose, "*.txt")))
wn_gelose, i_gelose, _ = traiter_acquisitions(liste_fichiers_gellose)

racine = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\exp_1\spectre_lumière_blanche"
fichiers = sorted(glob.glob(os.path.join(racine, '*.txt')))
wn_ref,_, intensite_ref_brute = traiter_acquisitions(fichiers)
t_lambda, lisse = caracteriser_motif_fixe(raman_shift_to_nm(wn_ref, 785), intensite_ref_brute)




def traiter_acquisitions_verre(liste_fichiers, traiter_etalon=True, als=True, bubblewidth=None, lam=1e6):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre du verre et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    
    #spectre sans rayon cosmiques
    wn, _, i = traiter_acquisitions(liste_fichiers)

    i_verre_corr = corriger_motif_fixe(raman_shift_to_nm(wn_verre, 785), i_verre, raman_shift_to_nm(wn_ref, 785), t_lambda)

    if traiter_etalon:
        #spectre sans rayon cosmiques et sans étalon
        i_corr_F = corriger_motif_fixe(raman_shift_to_nm(wn, 785), i, raman_shift_to_nm(wn_ref, 785), t_lambda)

        if als==True:
            #spectre sans rayon cosmiques, sans étalon et sans fluorescence
            i_corr_SF = corriger_fluorescence_als(i_corr_F, lam=lam)
        else:
            i_corr_SF = corriger_fluorescence(i_corr_F, min_bubble_widths=bubblewidth)

        #spectre sans rayon cosmiques, sans étalon, sans fluorescence et sans verre
        intensite = soustraire_spectre(wn, i_corr_SF, wn_verre, i_verre_corr)

    else:
        if als ==True:
            i_SF = corriger_fluorescence_als(i)

        else:
            i_SF = corriger_fluorescence(i)
        intensite = soustraire_spectre(wn, i_SF, wn_verre, i_verre_corr)

    
    intensite_centree = intensite - np.mean(intensite)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml



# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DE LA GELLOSE + CENTRAGE DES DONNÉES: JOUR 2 ET 4
# ────────────────────────────────────────────────────────────────────────


def traiter_acquisitions_gellose(liste_fichiers, traiter_etalon=True, als=True, bubblewidth=None, lam=1e6):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre du verre et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    
    #spectre sans rayon cosmiques
    wn, _, i = traiter_acquisitions(liste_fichiers)

    i_gelose_corr = corriger_motif_fixe(raman_shift_to_nm(wn_gelose, 785), i_gelose, raman_shift_to_nm(wn_ref, 785), t_lambda)


    if traiter_etalon:
        #spectre sans rayon cosmiques et sans étalon
        i_corr_F = corriger_motif_fixe(raman_shift_to_nm(wn, 785), i, raman_shift_to_nm(wn_ref, 785), t_lambda)

        if als==True:
            #spectre sans rayon cosmiques, sans étalon et sans fluorescence
            i_corr_SF = corriger_fluorescence_als(i_corr_F, lam=lam)
        else:
            i_corr_SF = corriger_fluorescence(i_corr_F, min_bubble_widths=bubblewidth)


        #spectre sans rayon cosmiques, sans étalon, sans fluorescence et sans verre
        intensite = soustraire_spectre(wn, i_corr_SF, wn_gelose, i_gelose_corr)

    else:
        if als ==True:
            i_SF = corriger_fluorescence_als(i)

        else:
            i_SF = corriger_fluorescence(i)
        #spectre sans rayon cosmiqueset et sans verre
        intensite = soustraire_spectre(wn, i_SF, wn_gelose, i_gelose)

    
    intensite_centree = intensite - np.mean(intensite)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml
#!!! a faire

def traiter_acquisitions_verre_gelose(liste_fichiers, retirer_etalon=True):

    wn, i, _ = traiter_acquisitions(liste_fichiers)
    wn_verre, i_verre, _ = traiter_acquisitions(liste_fichiers_verre)
    

    intensite_SV = soustraire_spectre(wn, i, wn_verre, i_verre)
    intensite_SV_SG = soustraire_spectre(wn, intensite_SV, wn_gelose, i_gelose)
    intensite_SV_SG_SF = corriger_fluorescence(intensite_SV_SG, min_bubble_widths=50, fit_order=1)

    intensite_centree = intensite_SV_SG_SF - np.mean(intensite_SV_SG_SF)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return raman_shift_to_nm(wn, 785), i_nrml

# ───────────────────────────────────────────────────
# on va créer des fonctions 
# pour extraire les fichiers par zone et non par 
# souris
#────────────────────────────────────────────────────



#─────────────────────────────────────1. FICHIER DES JOURS 8 ET 11───────────────────────────────────────────

racine1 = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya"

def extraire_fichiers_jours_8_11(jour, petri, souris, zone):
    dossier = os.path.join(racine1, jour, "Raman", petri, souris, zone)
            # Si le dossier n'existe pas, on le saute sans buguer
    if not os.path.exists(dossier):
        return []
    # Chercher les fichiers .txt dans ce dossier
    fichiers_zone = glob.glob(os.path.join(dossier, "*.txt"))
    #print(f'Premier 10 fichiers de la zone {zone} du jour {jour} : {fichiers_zone}')
    return fichiers_zone

#extraire_fichiers_jours_8_11("jour_8", "petri1", "souris1", "zone1")

#───────────────────────────────────────────2. FICHIER DU JOUR 2 ────────────────────────────────────────────

racine2 = r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya"

def extraire_fichiers_jour_2(jour, petri, souris, zone):
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
    
#extraire_fichiers_jour_2("jour2", "petri1", "souris1", "zone1")

#────────────────────────────────────────3. FICHIER DU JOUR 4 ──────────────────────────────────────────────

racine3 = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya"

def extraire_fichiers_jour_4(jour, petri, souris, zone, fichiers_par_zone=10):
    """
    Sépare les fichiers d'un dossier en zones selon l'ordre chronologique.
    Les 10 premiers (par date) = zone1, les 10 suivants = zone2, etc.
    
    zone : int (1, 2, 3...)
    """
    dossier = os.path.join(racine3, jour, "Raman", petri)

    pattern = os.path.join(dossier, f"{souris}*.txt")

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
    
    
    #print(f"{zone} — {len(fichiers_zone)} fichiers trouvés")
    return fichiers_zone

def extraire_fichiers_jour_0(jour, petri, souris, echantillon, zone, fichiers_par_zone=10):

    dossier = os.path.join(racine1, jour, 'raman', petri, souris)

    pattern = os.path.join(dossier, f"{echantillon}*.txt")
    tous_les_fichiers = sorted(glob.glob(pattern))
    
    if not tous_les_fichiers:
        print(f"Aucun fichier trouvé avec le pattern : {pattern}")
        return []
    
    # Trie par date de modification (le plus ancien en premier)
    tous_les_fichiers_tries = sorted(tous_les_fichiers, key=lambda f: os.path.getmtime(f))

        # Découpe en tranches de 10
    indice = int(zone[-1])
    debut = (indice - 1) * fichiers_par_zone
    fin = debut + fichiers_par_zone
    fichiers_zone = tous_les_fichiers_tries[debut:fin]

    #print(f"{zone} — {len(fichiers_zone)} fichiers trouvés")
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


def extraire_fichiers_jour8_frais(jour, petri, souris, zone):
    dossier = os.path.join(racine2, jour, "Raman", petri)
    pattern = os.path.join(dossier, f"*{souris}*{zone}*")

    dossiers_trouves = sorted(glob.glob(pattern))
    
    if not dossiers_trouves:
        print(f"Pour le {jour} La {zone} de la {souris} du {petri} n'existe pas")
        return []

    fichiers_zone = sorted(glob.glob(os.path.join(dossiers_trouves[0], "*.txt")))
    #print(f'Fichiers de la {zone} du {petri} du {jour} : {fichiers_zone}')
    
    return fichiers_zone

    return



extraire_fichiers_jour8_frais("Jour8", "petri1", "souris1", 'zone2')
