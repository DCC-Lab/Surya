"""
Analyse LDA des spectres Raman — gélose.

Deux analyses INDÉPENDANTES :
  - Dose seule       (0gy vs 45gy), peu importe le traitement
  - Traitement seul  (NT vs +P),    peu importe la dose

Chaque échantillon (souris) fournit 3 spectres (zones z1/z2/z3) qui restent
groupés lors de la validation croisée : LeaveOneGroupOut est fait par souris
(groups=souris_id), donc les 3 zones d'une même souris ne se retrouvent
jamais séparées entre train et test.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from sklearn.pipeline import Pipeline


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, balanced_accuracy_score, ConfusionMatrixDisplay

from extract_data import adjust_spectrum, extract_frais, extract_fixe, lecteur_données_moy_frais, lecteur_données_moy_fixe, charger_nocif, soustraire_spectre, lecteur_gelose

lecteurs = {
    'frais':extract_frais,
    'fixe':extract_fixe,
    'moyenfrais':lecteur_données_moy_frais,
    'moyenfixe': lecteur_données_moy_fixe
}




# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
from config import CONFIG


N_MAX_COMPOSANTES = 11  # borne supérieure explorée par le test de sélection



# ────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ────────────────────────────────────────────────────────────────────────────
def charger_spectres(config, etat):
    """Charge tous les spectres et les étiquettes
    """
    spectres, etiquettes = [], []
    w = None

    for batch, petris in config.items():
        for petri, (echantillon, dose, type_) in petris.items():
            if 'moyenne' in etat:
                a_lire = [(None, lecteurs[etat](batch, petri))]
            else:
                a_lire = [(z, lecteurs[etat](batch, petri, z)) for z in ['z1', 'z2', 'z3']]

            for zone, liste_fichiers in a_lire:
                if not liste_fichiers:
                    continue

                w_local, i = adjust_spectrum(liste_fichiers, soustraire_spectre=True)
                if w_local is None or i is None:
                    continue
                if not np.isfinite(i).all():
                    print(f"NaN/Inf : {echantillon} {zone or ''}, {petri}, {batch} — ignoré")
                    continue

                w = w_local
                suffixe = f"_{zone}" if zone else ""
                spectres.append(i)
                etiquettes.append(f"{echantillon}{suffixe}_{dose}{type_}_{etat}")

    return np.array(spectres), etiquettes, w



def parser_etiquettes(etiquettes):
    """Extrait échantillon, dose, sexe, traitement, id souris, état (frais/fixe), zone."""
    echantillons, doses, sexes, traitements, souris_id, etats, zones = [], [], [], [], [], [], []

    for e in etiquettes:
        parts = e.split('_')
        echantillon = parts[0]
        reste = parts[-2]   # ex: "45FNT"
        etat = parts[-1]    # "frais" ou "fixe"
        zone = parts[1] if len(parts) == 4 else None  # présent seulement si moyenne=False

        m = re.match(r'(\d+)([A-Z])(.*)', reste)
        dose, sexe, traitement = int(m.group(1)), m.group(2), m.group(3)
        mouse = echantillon.split('-')[0]

        echantillons.append(echantillon)
        doses.append(dose)
        sexes.append(sexe)
        traitements.append(traitement)
        souris_id.append(mouse)
        etats.append(etat)
        zones.append(zone)

    return (
        np.array(echantillons), np.array(doses),
        np.array(sexes), np.array(traitements),
        np.array(souris_id), np.array(etats),
        np.array(zones, dtype=object),
    )



def choisir_n_composantes(X, y, groupes, n_max=8, titre="Choix du nombre de composantes"):
    """Balaie le nombre de composantes PCA et évalue la balanced accuracy en
    validation croisée LeaveOneGroupOut, pour aider à choisir combien en
    garder avant le LDA final.
    Ce que fait la fonction
Objectif : trouver combien de composantes PCA garder avant d'entraîner un LDA final, en testant plusieurs valeurs et en comparant leurs performances.
Étape par étape :

valeurs_n = list(range(1, n_max + 1)) → elle va tester 1 composante, puis 2, puis 3... jusqu'à n_max (11 par défaut).
Pour chaque valeur n de composantes :

elle construit un pipeline : d'abord une PCA qui réduit X à n dimensions, puis un LDA (Linear Discriminant Analysis) qui classe les données à partir de ces n dimensions.
elle évalue ce pipeline avec cross_val_predict et une validation croisée LeaveOneGroupOut (LOGO) : à chaque tour, un groupe entier (par exemple un sujet, une session...) est mis de côté comme test, et le modèle est entraîné sur tous les autres groupes. Ça évite les fuites de données si plusieurs échantillons viennent de la même source.
elle calcule le balanced accuracy score entre les vraies étiquettes y et les prédictions, et stocke ce score.


Elle trace un graphique : score en fonction du nombre de composantes, avec une ligne horizontale à 0.5 (niveau du hasard pour un problème à 2 classes).
Elle retourne le n qui donne le meilleur score, et l'affiche.

En résumé : c'est une recherche du nombre optimal de composantes PCA, en observant à partir de combien de composantes l'ajout de dimensions supplémentaires n'améliore plus (ou dégrade) la performance de classification.
"""
    #n_max = min(n_max, X.shape[0] - 1, X.shape[1])
    valeurs_n = list(range(1, n_max + 1))
    scores = []
    logo = LeaveOneGroupOut()

    for n in valeurs_n:
        pipe = Pipeline([
            ('pca', PCA(n_components=n)),
            ('lda', LinearDiscriminantAnalysis()),
        ])
        y_pred = cross_val_predict(pipe, X, y, groups=groupes, cv=logo)
        scores.append(balanced_accuracy_score(y, y_pred))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valeurs_n, scores, marker='o')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0.5, color='grey', lw=0.5, linestyle='--', label='hasard (2 classes)')
    ax.set_xlabel("Nombre de composantes PCA")
    ax.set_ylabel("Balanced accuracy (CV LeaveOneGroupOut)")
    ax.set_title(titre)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    meilleur_n = valeurs_n[int(np.argmax(scores))]
    print(f"{titre} — meilleur score : {max(scores):.1%} avec {meilleur_n} composante(s)\n")
    return meilleur_n



def evaluer_lda(X_sub, y, groupes, n_pca, titre):
    """Pipeline PCA+LDA, entièrement re-fit à chaque pli LeaveOneGroupOut."""
    pipe = Pipeline([
        ('pca', PCA(n_components=n_pca)),
        ('lda', LinearDiscriminantAnalysis()),
    ])
    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(pipe, X_sub, y, groups=groupes, cv=logo)

    ba = balanced_accuracy_score(y, y_pred)
    print(f"── {titre} ──")
    print(classification_report(y, y_pred))
    print(f"Balanced accuracy : {ba:.1%}\n")

    return y_pred, ba


def entrainer_lda(X, y_labels, souris_id, masque, titre_suffixe, n_max=N_MAX_COMPOSANTES):
    """
    Entraîne un LDA (2 classes) discriminant sur les échantillons sélectionnés
    par `masque`, avec y_labels comme étiquette de classe.

    y_labels : array de même longueur que X, contenant l'étiquette à discriminer
               (ex: '0gy'/'45gy' pour la dose, 'NT'/'+P' pour le traitement)
    masque   : booléen, sert à sélectionner quels échantillons entrent dans le
               fit — c'est ici qu'on retire le groupe irradié+pansement.
    """
    X_sub = X[masque]
    y_sub = y_labels[masque]
    groupes_sub = souris_id[masque]

    n_pca = choisir_n_composantes(X_sub, y_sub, groupes_sub, n_max,
                                   f"Choix N_PCA — {titre_suffixe}")

    y_pred, ba = evaluer_lda(X_sub, y_sub, groupes_sub, n_pca, titre_suffixe)

    # PCA/LDA "finales" fit sur tout X_sub — servent à reconstruire le LD1
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_sub)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_pca, y_sub)

    return y_sub, y_pred, ba, n_pca, X_lda[:, 0], pca, lda






from scipy.signal import find_peaks
def annoter_pics(ax, w, spectre, n_pics=5, couleur='black'):
    """Détecte les n_pics plus grands pics (en valeur absolue) et les annote
    avec leur position en cm-1."""
    # on cherche les pics positifs ET négatifs séparément
    idx_pos, _ = find_peaks(spectre)
    idx_neg, _ = find_peaks(-spectre)
    idx_tous = np.concatenate([idx_pos, idx_neg])

    # on garde les n_pics avec la plus grande amplitude absolue
    amplitudes = np.abs(spectre[idx_tous])
    ordre = np.argsort(amplitudes)[::-1][:n_pics]
    idx_principaux = idx_tous[ordre]

    for idx in idx_principaux:
        x, y = w[idx], spectre[idx]
        ax.annotate(
            f"{x:.0f}",
            xy=(x, y),
            xytext=(0, 10 if y >= 0 else -15),  # décale le texte au-dessus/dessous
            textcoords='offset points',
            ha='center', fontsize=7, color=couleur,
            rotation=90,
            arrowprops=dict(arrowstyle='-', lw=0.5, color=couleur),
        )


def marquer_positions(ax, w, spectre, positions, couleur='red', 
                       ligne_verticale=True, annoter=True, fontsize=8):
    """
    Marque des positions spécifiques (en cm-1) sur un spectre :
    - trouve la valeur y du spectre à cette position (par interpolation)
    - trace un point à l'intersection
    - optionnellement une ligne verticale pointillée jusqu'au point
    - optionnellement une étiquette avec la valeur en cm-1

    positions : liste de nombres d'onde, ex. [1094, 1450, 1655]
    """
    for x_cible in positions:
        y_cible = np.interp(x_cible, w, spectre)  # interpolation linéaire

        # ligne verticale du bas du graphique jusqu'au point
        if ligne_verticale:
            ax.plot([x_cible, x_cible], [0, y_cible],
                     color=couleur, lw=0.8, linestyle='--', alpha=0.7)

        # point à l'intersection
        ax.plot(x_cible, y_cible, marker='o', color=couleur,
                 markersize=5, zorder=5)

        # étiquette
        if annoter:
            ax.annotate(
                f"{x_cible:.0f}",
                xy=(x_cible, y_cible),
                xytext=(0, 8 if y_cible >= 0 else -12),
                textcoords='offset points',
                ha='center', fontsize=fontsize, color=couleur,
            )

def etiquette_courte(id_souris, zone):
    """'S39' + 'z1' -> '39-z1' (retire le préfixe de lettres, ex: le 'S')."""
    num = re.sub(r'^\D+', '', str(id_souris))
    return f"{num}-{zone}" if zone else num


# ── Chargement des deux états ──────────────────────────────────────────────
X_frais, etiquettes_frais, w_frais = charger_spectres(CONFIG, 'frais')
X_fixe, etiquettes_fixe, w_fixe = charger_spectres(CONFIG, 'fixe')

X = np.concatenate([X_frais, X_fixe], axis=0)
etiquettes = etiquettes_frais + etiquettes_fixe   # ce sont des listes Python, "+" les concatène
w = w_frais   # en supposant que w_frais == w_fixe (mêmes wavenumbers pour les deux états)

echantillons, doses, sexes, traitements, souris_id, etats, zones = parser_etiquettes(etiquettes)

y_dose = np.array([f"{d}gy" for d in doses])

# À définir une seule fois, en dehors de la fonction (au niveau du module)
_fig_ld1, _ax_ld1 = None, None


def matrice_confusion(masque, discriminant, titre, graph=False, couleur=None, nouvelle_figure_ld1=False):
    global _fig_ld1, _ax_ld1

    if discriminant == 'dose':
        y_labels = y_dose
        ordre_classes = ["0gy", "45gy"]
        display_labels = ["Non-irradiated", "Irradiated"]
    else:
        y_labels = traitements
        ordre_classes = ["+P", "NT"]
        display_labels = ["Pansement", "Non traité"]

    y, y_pred, ba, n_pca, ld1, pca, lda = entrainer_lda(X, y_labels, souris_id, masque, titre)

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, ax=ax1,
        labels=ordre_classes,
        colorbar=True,
        normalize='true',
        im_kw={'vmin': 0, 'vmax': 1},
        cmap='RdPu',
        display_labels=display_labels
    )
    ax1.set_title(f" {titre} - Effet {discriminant} - {masque} ({n_pca} comp., BA={ba:.1%})")
    plt.tight_layout()
    plt.show()

    disc = None
    if graph:
        # Reconstruire le spectre discriminant LD1
        disc = pca.components_.T @ lda.scalings_[:, 0]
        disc = disc / np.linalg.norm(disc)

        # Créer une nouvelle figure LD1 seulement si on n'en a pas déjà une
        # (ou si l'utilisateur le demande explicitement)
        if nouvelle_figure_ld1 or _ax_ld1 is None:
            _fig_ld1, _ax_ld1 = plt.subplots(figsize=(11, 6))
            _ax_ld1.axhline(0, color='grey', lw=0.5)
            _ax_ld1.set_xlabel("Raman shift (cm⁻¹)")
            _ax_ld1.set_ylabel("LD1 weight")
            _ax_ld1.set_title("Spectres discriminants LD1")

        _ax_ld1.plot(w, disc, label=titre, color=couleur, lw=1.2)
        _ax_ld1.legend()

    return y, y_pred, ba, n_pca, ld1, pca, lda, disc


def afficher_ld1():
    """À appeler une fois que tous les spectres voulus ont été ajoutés."""
    if _fig_ld1 is not None:
        plt.figure(_fig_ld1.number)
        plt.tight_layout()
        plt.show()




# ════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Matrice de confusion (avec colorbar)
# ════════════════════════════════════════════════════════════════════════



def score_ld1(spectres, pca, lda):
    """Projette des spectres bruts sur un LD1 déjà figé (pca+lda entraînés sur NT).
    Le signe indique la classe prédite, le seuil de décision est à 0."""
    X_pca = pca.transform(spectres)
    return lda.decision_function(X_pca)

# ── Masques pour vos échantillons +P (jamais vus par ce LDA) ────────────────
#masque_PFi_0  = (etats == 'frais') & (traitements == '+P') & (doses == 0)
#masque_PFi_45 = (etats == 'frais') & (traitements == '+P') & (doses == 45)
#masque_NTFi_45 = (etats == 'frais') & (traitements == 'NT') & (doses == 45)

#scores_P_0gy  = score_ld1(X[masque_PFi_0],  pca_NTFi, lda_NTFi)
#scores_P_45gy = score_ld1(X[masque_PFi_45], pca_NTFi, lda_NTFi)
#scores_NT_45gy = score_ld1(X[masque_NTFi_45], pca_NTFi, lda_NTFi)

#pred_P_0gy  = np.where(scores_P_0gy  > 0, "45gy", "0gy")
#pred_P_45gy = np.where(scores_P_45gy > 0, "45gy", "0gy")
#pred_NT_45gy = np.where(scores_NT_45gy > 0, "45gy", "0gy")

#labels_P_0gy = [etiquette_courte(s, z) for s, z in zip(souris_id[masque_PFi_0], zones[masque_PFi_0])]
#labels_P_45gy = [etiquette_courte(s, z) for s, z in zip(souris_id[masque_PFi_45], zones[masque_PFi_45])]
#labels_NT_45gy = [etiquette_courte(s, z) for s, z in zip(souris_id[masque_NTFi_45], zones[masque_NTFi_45])]

#print("+P, vrai 0gy  → prédictions :", pred_P_0gy)
#print("+P, vrai 45gy → prédictions :", pred_P_45gy)




# ════════════════════════════════════════════════════════════════════════
# FIGURE — Projection 2D sur les axes LD1 (irradiation) et LD1 (pansement)
# ════════════════════════════════════════════════════════════════════════

# 4 groupes croisés dose × traitement (frais uniquement)
#groupes_2d = [
#    ((etats == 'frais') & (doses == 0)  & (traitements == 'NT'), 'tab:blue',    '0gy — NT'),
#    ((etats == 'frais') & (doses == 45) & (traitements == 'NT'), 'xkcd:scarlet','45gy — NT'),
#    ((etats == 'frais') & (doses == 0)  & (traitements == '+P'), 'tab:cyan',    '0gy — +P'),
#    ((etats == 'frais') & (doses == 45) & (traitements == '+P'), 'tab:orange',  '45gy — +P'),
#]

#fig, ax = plt.subplots(figsize=(9, 7))

#for masque, couleur, label in groupes_2d:
#    if not masque.any():
#        continue

#    x_vals = score_ld1(X[masque], pca_dose, lda_dose)
#    y_vals = score_ld1(X[masque], pca_pans, lda_pans)
#    labels_pts = [etiquette_courte(s, z) for s, z in zip(souris_id[masque], zones[masque])]

#    ax.scatter(x_vals, y_vals, color=couleur, s=60, alpha=0.85,
#               edgecolor='k', label=label, zorder=3)

#    for x, y, lbl in zip(x_vals, y_vals, labels_pts):
#        ax.annotate(lbl, xy=(x, y), xytext=(0, 8), textcoords='offset points',
#                    ha='center', fontsize=7, color=couleur)

#ax.axhline(0, color='grey', linestyle='--', lw=1)
#ax.axvline(0, color='grey', linestyle='--', lw=1)
#ax.set_xlabel("Score LD1 — Irradiation (0gy vs 45gy)")
#ax.set_ylabel("Score LD1 — Pansement (NT vs +P)")
#ax.set_title("Projection des spectres sur les axes irradiation × pansement")
#ax.legend(loc='best', fontsize=8)
#plt.tight_layout()
#plt.show()






