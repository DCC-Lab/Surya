import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from orpl.baseline_removal import bubblefill
from scipy.optimize import lsq_linear
from scipy import sparse
from scipy.sparse.linalg import spsolve



# ─────────────────────────────────────────────
# RACINES
# ─────────────────────────────────────────────

# root_local = Path(r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya")
root_local = Path(r"/Volumes/Goliath/dcclab/surya")
# root_local = Path(r"/Users/dccote/surya")
root_cafeine = Path(r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya")



# dir_exp1 = root_local / "exp_1"
# dir_exp2 = root_local / "exp_2"
# dir_exp1_frais = root_local / "exp_2"



racine1 = root_cafeine / "exp_1"
racine2 = root_cafeine / "exp_2"
racine3 = r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya\exp_1"

# ─────────────────────────────────────────────
# EXP#1 - EXTRACT FILE LIST
# ─────────────────────────────────────────────
def extract_jour0(petri, souris, zone, fichiers_par_zone=10):

    dossier = os.path.join(racine1, 'jour0', 'fixe', 'raman', petri, souris)
    tous_les_fichiers = sorted(glob.glob(dossier))
    
    if not tous_les_fichiers:
        print(f"Aucun fichier trouvé avec le pattern : {dossier}")
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



def extract_jour2(petri, souris, zone, matiere='verre', fichiers_par_zone=10): #matiere = gelose ou verre

    dossier = os.path.join(root_cafeine,'exp_1', 'jour2', "raman", 'fixe',  petri)
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
    print(f"{zone} — {len(fichiers_zone)} fichiers trouvés: {fichiers_zone}")
    
    return fichiers_zone

#fichiers = extract_jour2('gelose', 'petri1', 'souris1', 'zone1')


def extract_jour4(petri, souris, zone, fichiers_par_zone=10):
    """
    Sépare les fichiers d'un dossier en zones selon l'ordre chronologique.
    Les 10 premiers (par date) = zone1, les 10 suivants = zone2, etc.
    
    zone : int (1, 2, 3...)
    """
    dossier = os.path.join(racine1, 'jour4', 'fixe', "raman", petri)

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

def extract_jour8_jour11(jour, petri, souris, zone):
    dossier = os.path.join(racine1, jour, 'fixe',  "Raman", petri, souris, zone)
            # Si le dossier n'existe pas, on le saute sans buguer
    if not os.path.exists(dossier):
        return []
    # Chercher les fichiers .txt dans ce dossier
    fichiers_zone = glob.glob(os.path.join(dossier, "*.txt"))
    #print(f'Premier 10 fichiers de la zone {zone} du jour {jour} : {fichiers_zone}')
    return fichiers_zone


# ─────────────────────────────────────────────
# EXP#2 - EXTRACT FILE LIST
# ─────────────────────────────────────────────

def extract_gelose(batch, petri):
    dossier = os.path.join(racine2, batch, 'frais', petri)
    pattern = os.path.join(dossier, f"*petri*")
    tous_les_fichiers = sorted(glob.glob(pattern))
    
    if not tous_les_fichiers:
        print(f"Aucun fichier trouvé avec le pattern : {pattern}")
        return []

    return tous_les_fichiers

def extract_frais(batch, petri, zone):
    dossier = os.path.join(racine2, batch, 'frais', petri)
    pattern = os.path.join(dossier, f"*{zone}*.txt")
    tous_les_fichiers = sorted(glob.glob(pattern))
    
    if not tous_les_fichiers:
        print(f"Aucun fichier trouvé avec le pattern : {pattern}")
        return []

    return tous_les_fichiers

def extract_fixe(batch, petri, zone):
    dossier = os.path.join(racine2, batch, 'fixe', petri)
    pattern = os.path.join(dossier, f"*{zone}*.txt")
    tous_les_fichiers = sorted(glob.glob(pattern))
    
    if not tous_les_fichiers:
        print(f"Aucun fichier trouvé avec le pattern : {pattern}")
        return []
    
    return tous_les_fichiers


def lecteur_données_moy_frais(batch, petri, zone):
    dossier = os.path.join(racine2, batch, 'frais', petri)
    pattern = os.path.join(dossier, f'*z*.txt')
    tous_les_fichiers= sorted(glob.glob(pattern))
    if not tous_les_fichiers:
        return []
    return tous_les_fichiers

def lecteur_données_moy_fixe(batch, petri, zone):
    dossier = os.path.join(racine2, batch, 'fixe', petri)
    pattern = os.path.join(dossier, f'*z*.txt')
    tous_les_fichiers= sorted(glob.glob(pattern))
    if not tous_les_fichiers:
        return []
    return tous_les_fichiers

# ──────────────────────────────────────────────────────────────────────────────────────────
# EXTRACT DATA FROM FILE
# ──────────────────────────────────────────────────────────────────────────────────────────

def formater_donnees(chemin_fichier, wn_min=500, wn_max=3200):
    integration = None
    data = []
    dans_les_donnees = False

    with open(chemin_fichier, 'r') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue

            if 'Integration Time' in ligne:
                valeur_str = ligne.split(':')[-1].strip().replace(',', '.')
                integration = float(valeur_str)
                continue

            if ligne.startswith('>>>>>Begin Spectral Data'):
                dans_les_donnees = True
                continue

            if not dans_les_donnees:
                continue  # on ignore tout ce qui précède le marqueur

            valeurs = [float(x) for x in ligne.replace(',', '.').split()]
            if len(valeurs) >= 2:
                data.append(valeurs[:2])

    if integration is None:
        raise ValueError(f"Temps d'intégration introuvable dans {chemin_fichier}")

    data = np.array(data)
    if data.ndim != 2 or data.shape[0] == 0:
        print(f"Fichier vide ou mal formaté : {chemin_fichier}")
        return None, None

    wn = data[:, 0]
    intensite = data[:, 1] / integration
    masque = (wn >= wn_min) & (wn <= wn_max)
    return wn[masque], intensite[masque]


# ──────────────────────────────────────────────────────────────────────────────────────────
# REMOVE COSMIC RAYS
# ──────────────────────────────────────────────────────────────────────────────────────────

def retirer_rayons_cosmiques(wn, intensite, seuil=6.0, fenetre=14, largeur_max=2, zones_protegees=None):
    """
    Détecte et remplace les spikes de rayons cosmiques.

    seuil       : sensibilité de détection (plus bas = plus sensible, plus de faux positifs possibles)
    fenetre     : taille du voisinage utilisé pour estimer la médiane/MAD locale
    largeur_max : largeur maximale (en points) d'une anomalie pour être considérée
                  comme un rayon cosmique plutôt qu'un vrai pic Raman.
                  Avec ~2.74 cm-1/point, largeur_max=2 couvre les spikes de 1-2 pixels
                  tout en protégeant les vrais pics Raman (FWHM typique >= 4-7 points).
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

    # ── Étape 1 : détecter tous les points candidats ──
    candidats = np.zeros(n, dtype=bool)
    for i in range(demi, n - demi):
        if masque_protege[i]:
            continue
        voisins = np.concatenate([intensite[i-demi:i], intensite[i+1:i+demi+1]])
        mediane = np.median(voisins)
        mad = np.median(np.abs(voisins - mediane)) + 1e-10
        if abs(intensite[i] - mediane) > seuil * mad:
            candidats[i] = True

    # ── Étape 2 : regrouper les candidats consécutifs en segments ──
    i = 0
    while i < n:
        if not candidats[i]:
            i += 1
            continue
        j = i
        while j < n and candidats[j]:
            j += 1
        largeur = j - i   # segment [i, j) de candidats consécutifs

        if largeur <= largeur_max:
            # spike ponctuel -> on corrige en interpolant entre les bords sains du segment
            gauche = i - 1
            droite = j
            if gauche >= 0 and droite < n:
                for k in range(i, j):
                    intensite_corr[k] = np.interp(k, [gauche, droite],
                                                   [intensite[gauche], intensite[droite]])
        # sinon (largeur > largeur_max) -> probablement un vrai pic, on ne touche pas
        i = j

    return intensite_corr


# ──────────────────────────────────────────────────────────────────────────────────────────
# AVERAGE DATA FROM FILE LIST
# ──────────────────────────────────────────────────────────────────────────────────────────

def traiter_acquisitions(liste_fichiers):
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
        
        intensite = retirer_rayons_cosmiques(wn_ref, intensite)


        # ajout à la liste des spectres
        spectres.append(intensite)

    # Moyennage des acquisitions : on a maintenant 1 spectre pour les 20 ou 30 acquisitions
    spectre_moyen = np.mean(spectres, axis=0)

    return wn_ref, spectre_moyen



# ──────────────────────────────────────────────────────────────────────────────────────────
# REMOVE STANDARDIZATION EFFECT
# ──────────────────────────────────────────────────────────────────────────────────────────

def raman_shift_to_nm(shift_cm1, laser_nm):
    nu_laser = 1e7 / laser_nm          # cm^-1
    nu_scattered = nu_laser - shift_cm1  # Stokes
    return 1e7 / nu_scattered           # nm

def caracteriser_motif_fixe(intensite_ref_brute=None, fenetre_lissage=101, ordre_poly=3, methode='savgol'):
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
    else:
        raise ValueError("methode doit être 'savgol' ou 'spline'")
 
    # Évite division par ~0
    lisse = np.where(np.abs(lisse) < 1e-9, 1e-9, lisse)
 
    t_lambda = intensite_ref_brute / lisse
    return t_lambda, lisse
 
 
def corriger_motif_fixe(intensite_echantillon, t_lambda):
    """
    Applique la correction de motif fixe à un spectre échantillon.
    Interpole t_lambda sur la grille de l'échantillon si nécessaire.
    """
    t_interp = t_lambda
 
    return intensite_echantillon / t_interp


# ──────────────────────────────────────────────────────────────────────────────────────────
# REMOVE FLUORESCENCE
# ──────────────────────────────────────────────────────────────────────────────────────────

def supprimer_fluorescence(intensite, min_bubble_widths=90, fit_order=1):
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



def supprimer_fluorescence_als(intensite, lam=1e6, p=0.01, n_iter=15):
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

    return intensite_corrigee, baseline


import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def build_lambda_vector(x, lam_low=1e4, lam_high=1e7, transition_center=2300, transition_width=100):
    """
    Construit un vecteur lambda qui vaut ~lam_low avant transition_center
    et transitionne graduellement vers lam_high apres, sur une largeur
    transition_width (en unites de x, ex: cm^-1).

    On travaille en log(lambda) parce que lam_low et lam_high sont sur
    des ordres de grandeur tres differents (1e4 vs 1e7) : une transition
    lineaire en lambda directement ferait un saut trop brusque au debut.

    transition_center : x ou on est a mi-chemin entre lam_low et lam_high
    transition_width   : plus petit = transition plus abrupte,
                          plus grand = transition plus douce
    """
    # Sigmoide entre 0 et 1
    s = 1 / (1 + np.exp(-(x - transition_center) / transition_width))

    log_lam = np.log10(lam_low) + s * (np.log10(lam_high) - np.log10(lam_low))
    lam_vector = 10 ** log_lam

    return lam_vector


def als_baseline_variable_lambda(y, lam_vector, p, niter=10):
    """
    lam_vector : array de meme longueur que y, la rigidite locale.
    Grand lam -> baseline tres lisse/rigide (colle moins aux pics).
    Petit lam -> baseline plus souple (suit plus les pics).
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    Lam = sparse.diags(lam_vector[1:-1])
    w = np.ones(L)
    for i in range(niter):
        W = sparse.diags(w)
        Z = W + D @ Lam @ D.T
        z = spsolve(Z.tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_with_lipid_protection(x, y, lam_low=1e4, lam_high=1e7,
                                     transition_center=2300, transition_width=100,
                                     p=0.01, niter=10):
    """
    Fonction pratique qui combine les deux etapes ci-dessus.
    """
    lam_vector = build_lambda_vector(x, lam_low, lam_high, transition_center, transition_width)
    z = als_baseline_variable_lambda(y, lam_vector, p=p, niter=niter)
    i_corr = y - z
    return i_corr




# ──────────────────────────────────────────────────────────────────────────────────────────
# AVERAGE DATA FROM FILE LIST + REMOVE STANDARDIZATION EFFECT + REMOVE FLUORESCENCE
# ──────────────────────────────────────────────────────────────────────────────────────────
print("Debut lecture donnees")
racine = root_cafeine / Path(r"exp_1/spectre_lumière_blanche")
print(racine)
fichiers = sorted(glob.glob(os.path.join(racine, '*.txt')))
print("Debut traitement acquisitions")
w_ref, i_ref = traiter_acquisitions(fichiers)
print("Debut caracteriser motif fixe")
t_lambda, lisse = caracteriser_motif_fixe(intensite_ref_brute=i_ref)
def correction_data(liste_fichiers, traiter_etalon=True, als=True, bubblewidth=None, lam=1e6, p=0.01):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre du verre et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    #spectre sans rayon cosmiques
    w, i = traiter_acquisitions(liste_fichiers)

    if traiter_etalon:
        #spectre sans rayon cosmiques et sans étalon
        i_corr_F = corriger_motif_fixe(i, t_lambda)
        if als==True:
            intensite, baseline = supprimer_fluorescence_als(i_corr_F, lam=lam, p=p)
            intensite = baseline_with_lipid_protection(w, intensite)
        else:
            intensite = supprimer_fluorescence(i_corr_F, min_bubble_widths=bubblewidth)
            intensite = baseline_with_lipid_protection(w, intensite)
    else:
        if als==True:
            intensite, baseline = supprimer_fluorescence_als(i, lam=lam, p=p)
            intensite = baseline_with_lipid_protection(w, intensite)
        else:
            intensite = supprimer_fluorescence(i, min_bubble_widths=bubblewidth)
            intensite = baseline_with_lipid_protection(w, intensite)
    
    return w, intensite


# ──────────────────────────────────────────────────────────────────────────────────────────
# REMOVE ROGUE SPECTRUM
# ──────────────────────────────────────────────────────────────────────────────────────────
# mettre à 0 tous mes spectres 
def soustraire_spectre1(wn_echantillon, intensite_echantillon, wn_nocif, intensite_nocif, ordre_baseline=1, fenetres_fit=None):
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
    
    wn_echantillon = np.asarray(wn_echantillon, dtype=float)
    intensite_echantillon = np.asarray(intensite_echantillon, dtype=float)
    wn_nocif = np.asarray(wn_nocif, dtype=float)
    intensite_nocif = np.asarray(intensite_nocif, dtype=float)

    # ... reste du code inchangé
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

from scipy.optimize import least_squares

def soustraire_spectre2(wn_echantillon, intensite_echantillon, wn_nocif, intensite_nocif,
                        ordre_baseline=1, fenetres_fit=None, robuste=True):
    wn_echantillon = np.asarray(wn_echantillon, dtype=float)
    intensite_echantillon = np.asarray(intensite_echantillon, dtype=float)
    nocif_interp = np.interp(wn_echantillon, wn_nocif, intensite_nocif)

    # ── Normalisation de la colonne du gel pour le conditionnement numérique ──
    echelle_nocif = np.std(nocif_interp)
    nocif_norm = nocif_interp / echelle_nocif

    if fenetres_fit is not None:
        masque = np.zeros_like(wn_echantillon, dtype=bool)
        for (lo, hi) in fenetres_fit:
            masque |= (wn_echantillon >= lo) & (wn_echantillon <= hi)
    else:
        masque = np.ones_like(wn_echantillon, dtype=bool)

    x_norm = (wn_echantillon - wn_echantillon.mean()) / wn_echantillon.std()
    colonnes = [nocif_norm] + [x_norm**k for k in range(ordre_baseline + 1)]
    A_full = np.column_stack(colonnes)
    A_fit = A_full[masque]
    y_fit = intensite_echantillon[masque]

    n_baseline = ordre_baseline + 1
    bornes_inf = [0.0] + [-np.inf] * n_baseline
    bornes_sup = [np.inf] + [np.inf] * n_baseline
    print("nb points dans le masque:", masque.sum())
    print("nocif_norm dans fenêtres:", nocif_norm[masque][:5])
    print("y_fit:", y_fit[:5])
    if robuste:
        def residuals(coeffs):
            return A_fit @ coeffs - y_fit
        x0 = np.zeros(A_fit.shape[1])
        resultat = least_squares(residuals, x0, bounds=(bornes_inf, bornes_sup),
                                  loss='soft_l1', f_scale=np.std(y_fit))
        coeffs = resultat.x
    else:
        resultat = lsq_linear(A_fit, y_fit, bounds=(bornes_inf, bornes_sup))
        coeffs = resultat.x

    alpha_norm = coeffs[0]
    alpha = alpha_norm / echelle_nocif   # ← alpha "réel", à appliquer sur nocif_interp brut

    modele_complet = A_full @ coeffs
    intensite_corrigee = intensite_echantillon - modele_complet

    return intensite_corrigee, alpha

def interpoler_zones_nocif(wn, intensite, zones_a_retirer):
    intensite = np.asarray(intensite, dtype=float).copy()
    wn = np.asarray(wn, dtype=float)

    for (lo, hi) in zones_a_retirer:
        masque = (wn >= lo) & (wn <= hi)
        if not masque.any():
            continue
        i_debut = np.where(masque)[0][0] - 1
        i_fin = np.where(masque)[0][-1] + 1
        if i_debut < 0 or i_fin >= len(wn):
            continue
        intensite[masque] = np.interp(wn[masque],
                                        [wn[i_debut], wn[i_fin]],
                                        [intensite[i_debut], intensite[i_fin]])
    return intensite


from scipy.optimize import minimize_scalar

def soustraire_reference_gel(spectre, gel_ref, bornes_alpha=(0, 2)):
    """
    Soustrait spectre_gel_ref * alpha du spectre échantillon,
    en optimisant alpha pour minimiser les résidus négatifs
    (signe qu'on a trop soustrait) tout en réduisant les pics du gel.
    """
    def cout(alpha):
        residu = spectre - alpha * gel_ref
        # pénalise fortement le sur-soustraction (résidus négatifs)
        penalite_neg = np.sum(np.clip(-residu, 0, None) ** 2)
        return penalite_neg

    res = minimize_scalar(cout, bounds=bornes_alpha, method='bounded')
    alpha_opt = res.x

    spectre_corrige = spectre - alpha_opt * gel_ref
    spectre_corrige = np.clip(spectre_corrige, 0, None)  # évite les négatifs résiduels

    return spectre_corrige, alpha_opt


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# AVERAGE DATA FROM FILE LIST + REMOVE STANDARDIZATION + REMOVE FLUORESCENCE + REMOVE ROGUE SPECTRUM + NORMALIZATION/CENTERING
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

from config import CONFIG1

def charger_nocif(config):
    i_s = []

    for batch, petris in config.items():
        for petri, (echantillon, dose, type_) in petris.items():
            fichiers = extract_gelose(batch, petri)
            if not fichiers:
                continue
            w, i= correction_data(fichiers)
  
            i_s.append(i)
            if not i_s:
                raise ValueError("Aucun spectre de gélose (nocif) n'a pu être chargé.")
            i_arr = np.array(i_s)
    #plt.plot(w, np.mean(i_arr, axis=0))
    #plt.show()
    return np.mean(i_arr, axis=0)

# i_arr_nocif = charger_nocif(CONFIG)

def adjust_spectrum(list_fich_echantillon, i_nocif=None, retirer_nocif=True, wn_min=600, wn_max=2800):

    if i_nocif is None:
        i_nocif = charger_nocif(CONFIG1)
    w, i = correction_data(list_fich_echantillon)
    if retirer_nocif:
        fenetres_gel = [
            (590, 650),
            (760, 820),
            (980, 1040),
            (1150, 1200),
            (1420, 1460),
            (1560, 1610),
            # (1280, 1340) exclue : corr ≈ 0.025, ça n'est pas du gel
        ]

        i_corr, alpha = soustraire_spectre2(
            w, i,
            w, i_nocif,
            ordre_baseline=1,
            fenetres_fit=fenetres_gel,   # ← ne pas oublier !
            robuste=True
        )
        print("alpha =", alpha)
        print('Début ploting')
        plt.plot(w, i, label='échantillon')
        plt.plot(w, i_corr, label='i_corr')
        plt.plot(w, i_nocif, label='nocif')
        plt.legend()
        plt.show()
    else:
        i_corr = i

    masque = (w >= wn_min) & (w <= wn_max)
    w_masque, i_masque = w[masque], i_corr[masque]

    intensite_centree = i_masque - np.mean(i_masque)
    i_nrml = intensite_centree / np.max(np.abs(intensite_centree))


    return w_masque, i_nrml


    










































# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DE LA GELLOSE + CENTRAGE DES DONNÉES: JOUR 2 ET 4
# ────────────────────────────────────────────────────────────────────────













def tester_als_settings(wn, i_corr_F, combos, wn_min=800, wn_max=2200):
    """
    Teste plusieurs combinaisons (lam, p) de corriger_fluorescence_als
    sur un même spectre, et affiche baseline + spectre corrigé pour chaque.

    combos : liste de tuples (lam, p), ex: [(1e5, 0.01), (1e6, 0.01), (1e7, 0.01), (1e6, 0.001)]
    """
    masque = (wn >= wn_min) & (wn <= wn_max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for lam, p in combos:
        corrige, baseline = supprimer_fluorescence_als(i_corr_F, lam=lam, p=p)
        axes[0].plot(wn[masque], i_corr_F[masque], color='grey', lw=0.5, alpha=0.5)
        axes[0].plot(wn[masque], baseline[masque], label=f"lam={lam:.0e}, p={p}")
        axes[1].plot(wn[masque], corrige[masque], label=f"lam={lam:.0e}, p={p}")

    axes[0].set_title("Baselines ALS superposées sur le spectre brut")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_title("Spectres corrigés résultants")
    axes[1].set_xlabel("Nombre d'onde (cm⁻¹)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    print("Debut lecture donnees")
    racine = root_cafeine / Path(r"exp_1/spectre_lumière_blanche")
    print(racine)
    fichiers = sorted(glob.glob(os.path.join(racine, '*.txt')))
    print("Debut traitement acquisitions")
    w_ref, i_ref = traiter_acquisitions(fichiers)

    print("Debut caracteriser motif fixe")
    t_lambda, lisse = caracteriser_motif_fixe(intensite_ref_brute=i_ref)
    # print("Starting code")

    fichiers = extract_frais('batch#1', 'petri5', 'z1')
    fichiers2 = extract_frais('batch#3', 'petri22', 'z1')
    print("Debut traitement acquistion")
    w1, i1 = correction_data(fichiers)
    w9, i9 = correction_data(fichiers2)

    #w, i_nrml, i, i_recu = adjust_spectrum(fichiers)

    print("Debut plotting")
    plt.plot(w1, i1, label='femelle irradié')
    plt.plot(w9, i9, label='male irradié')
    #plt.plot(w, i, label='spectre brut')
    #plt.plot(w, i, label='i apres suppression nocif (soustraire spectre 2)')



    plt.legend()
    plt.show()
