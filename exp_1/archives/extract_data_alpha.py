import numpy as np
import os
from orpl.baseline_removal import bubblefill
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import nnls
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


from scipy.optimize import lsq_linear

def soustraire_spectre(wn_echantillon, intensite_echantillon, 
                        wn_nocif, intensite_nocif,
                        ordre_baseline=1, fenetres_fit=None):
    nocif_interp = np.interp(wn_echantillon, wn_nocif, intensite_nocif)

    if fenetres_fit is not None:
        masque = np.zeros_like(wn_echantillon, dtype=bool)
        for (lo, hi) in fenetres_fit:
            masque |= (wn_echantillon >= lo) & (wn_echantillon <= hi)
    else:
        masque = np.ones_like(wn_echantillon, dtype=bool)

    x_norm = (wn_echantillon - wn_echantillon.mean()) / wn_echantillon.std()
    colonnes = [nocif_interp] + [x_norm**k for k in range(ordre_baseline + 1)]
    A_full = np.column_stack(colonnes)

    A_fit = A_full[masque]
    y_fit = intensite_echantillon[masque]

    n_baseline = ordre_baseline + 1
    bornes_inf = [0.0] + [-np.inf] * n_baseline
    bornes_sup = [np.inf] + [np.inf] * n_baseline

    resultat = lsq_linear(A_fit, y_fit, bounds=(bornes_inf, bornes_sup))
    coeffs = resultat.x
    alpha = coeffs[0]

    modele_complet = A_full @ coeffs
    intensite_corrigee = intensite_echantillon - modele_complet

    return intensite_corrigee, alpha   # ← retourne maintenant alpha aussi

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


    return wn_ref, spectre_moyen





# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DU VERRE + CENTRAGE DES DONNÉES: JOUR 8 ET 11
# ────────────────────────────────────────────────────────────────────────


dossier_verre = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier_verre, "*.txt")))


def traiter_acquisitions_verre(liste_fichiers, wn_min=500, wn_max=3025, retirer_cosmiques=True):
    wn, i = traiter_acquisitions(liste_fichiers, wn_min, wn_max, retirer_cosmiques)
    wn_verre, i_verre = traiter_acquisitions(liste_fichiers_verre, wn_min, wn_max, retirer_cosmiques)
    
    if wn is None or i is None or wn_verre is None or i_verre is None:
        return None, None, None

    intensite_SV, alpha = soustraire_spectre(wn, i, wn_verre, i_verre)
    intensité_SV_SF = corriger_fluorescence(intensite_SV, min_bubble_widths=50, fit_order=1)

    intensite_centree = intensité_SV_SF - np.mean(intensité_SV_SF)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml, alpha   # ← alpha en plus

# ────────────────────────────────────────────────────────────────────────
# 6. RETRAITS DE LA GELLOSE + CENTRAGE DES DONNÉES: JOUR 2 ET 4
# ────────────────────────────────────────────────────────────────────────

dossier_gellose = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\spectre_gellose"
liste_fichiers_gellose = sorted(glob.glob(os.path.join(dossier_gellose, "*.txt")))

def traiter_acquisitions_gellose(liste_fichiers, wn_min=500, wn_max=3025, retirer_cosmiques=True):
    """
    Traite une liste de fichiers .txt 20 ou 30 acquisitions (10 acquisitions par zones).
    Soustrait le spectre de la gellose et corrige la fluorescence.
    Centrage des données en soustrayant la moyenne.
    Retourne (wavenumbers, spectre_centré).
    """
    wn, i = traiter_acquisitions(liste_fichiers, wn_min, wn_max, retirer_cosmiques)
    wn_gellose, i_gellose = traiter_acquisitions(liste_fichiers_gellose, wn_min, wn_max, retirer_cosmiques)
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
    intensite_SG, alpha= soustraire_spectre(wn, i, wn_gellose, i_gellose)
    intensité_SG_SF = corriger_fluorescence(intensite_SG, min_bubble_widths=50, fit_order=1)

    intensite_centree = intensité_SG_SF - np.mean(intensité_SG_SF)
    i_nrml = intensite_centree / np.max(intensite_centree)
    
    return wn, i_nrml, alpha


# ─────────────────────────────────────────────
# Fonction qui évalue le coefficient alpha
# ─────────────────────────────────────────────

def filtrer_spectres_par_alpha(dict_spectres, seuil_mad=3.0, min_echantillons=4):
    """
    dict_spectres : dict {nom_echantillon: (wn, intensite, alpha)}
    seuil_mad     : nombre de MAD au-delà duquel un alpha est jugé aberrant
    min_echantillons : nombre minimal d'échantillons requis pour filtrer ce groupe.
                        En dessous, on garde tout (pas assez de données pour juger).

    Retourne (dict_spectres_filtrés, dict_spectres_rejetés)
    """
    noms = list(dict_spectres.keys())
    
    if len(noms) < min_echantillons:
        print(f"⚠️ Seulement {len(noms)} échantillon(s) — pas de filtrage (minimum {min_echantillons} requis)\n")
        return dict_spectres, {}

    alphas = np.array([dict_spectres[nom][2] for nom in noms])

    mediane = np.median(alphas)
    mad = np.median(np.abs(alphas - mediane)) + 1e-10

    spectres_ok = {}
    spectres_rejetes = {}

    for nom, alpha in zip(noms, alphas):
        ecart = abs(alpha - mediane) / mad
        if ecart > seuil_mad:
            spectres_rejetes[nom] = alpha
        else:
            spectres_ok[nom] = dict_spectres[nom]

    print(f"Médiane α : {mediane:.4f} | MAD : {mad:.4f} | seuil : {seuil_mad} MAD")
    print(f"{len(spectres_ok)} conservés, {len(spectres_rejetes)} rejetés")
    if spectres_rejetes:
        for nom, alpha in sorted(spectres_rejetes.items(), key=lambda x: -x[1]):
            print(f"  ❌ {nom} : α = {alpha:.4f}")
    print()

    return spectres_ok, spectres_rejetes


def filtrer_par_groupe(resultats, cle_groupe_fn, seuil_mad=3.0, min_echantillons=4):
    """
    Groupe les résultats selon cle_groupe_fn(nom_etiquette) puis filtre chaque groupe séparément.
    
    cle_groupe_fn : fonction qui prend une étiquette et retourne la clé de groupe (ex: le jour)
    """
    groupes = {}
    for nom, valeurs in resultats.items():
        cle = cle_groupe_fn(nom)
        groupes.setdefault(cle, {})[nom] = valeurs

    tous_ok = {}
    tous_rejetes = {}

    for cle, groupe in groupes.items():
        print(f"── Groupe : {cle} ({len(groupe)} échantillons) ──")
        ok, rejetes = filtrer_spectres_par_alpha(groupe, seuil_mad=seuil_mad, min_echantillons=min_echantillons)
        tous_ok.update(ok)
        tous_rejetes.update(rejetes)

    return tous_ok, tous_rejetes




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
racine3 = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya"

def lecteur_fichier_j4(jour, petri, souris):
    ''' 
    Gère la structure : racine/jour/raman/petri/souris_dose_zone*/
    cependant, les souris sont mélangées en un dossier
    '''
    dossier_petri = os.path.join(racine3, jour, "raman", petri)

    if not os.path.exists(dossier_petri):
        print(f"Dossier absent : {dossier_petri}")
        return []   # ← retourne liste vide mais l'appelant continue

    # cherche tous les fichiers qui commencent par le nom de la souris
    pattern = os.path.join(dossier_petri, f"{souris}*.txt")
    fichiers = sorted(glob.glob(pattern))

    return fichiers

#w, i = traiter_acquisitions_j2_j4(lecteur_fichier_j4("jour4", "petri1", "souris4"), wn_min=500, wn_max=3025, retirer_cosmiques=True)
#print(f"première 10 longueurs d'onde : {w[:10]}")
#print(f"première 10 intensités : {i[:10]}")

def lecteur_fichier_j0(jour, petri, souris, échantillon):

    dossier = os.path.join(racine1, jour, "raman", petri, souris)

    if not os.path.exists(dossier):
        print(f"Dossier absent : {dossier}")
        return []     
    
    pattern = os.path.join(dossier, f"{échantillon}*.txt")
    fichiers = sorted(glob.glob(pattern))
    #print(f'premier 12 fichiers : {fichiers[:12]}')
    
    return fichiers

#lecteur_fichier_j0("jour0", "petri1", "souris1", "echantillon1")