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

from extract_data import traiter_acquisitions_gellose, lecteur_données_frais, lecteur_données_fixes, lecteur_données_moy


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'batch#1': {
        'petri1':  ('S48-G', 45, 'FNT'),
        'petri2':  ('S48-D', 0,  'FNT'),
        'petri3':  ('S38-G', 45, 'FNT'),
        'petri4':  ('S38-D', 0,  'FNT'),
        'petri5':  ('S40-G', 45, 'FNT'),
        'petri6':  ('S40-D', 0,  'FNT'),
        'petri7':  ('S47-G', 45, 'FNT'),
        'petri8':  ('S47-D', 0,  'FNT'),
        # 'petri9':  ('S39-G', 0,  'FNT'),
        # 'petri10': ('S39-D', 0,  'FNT'),
    },
    'batch#2': {
        'petri11': ('S45-G', 45, 'F+P'),
        'petri12': ('S45-D', 0,  'F+P'),
        'petri13': ('S41-G', 45, 'F+P'),
        'petri14': ('S41-D', 0,  'F+P'),
        'petri15': ('S42-G', 45, 'F+P'),
        'petri16': ('S42-D', 0,  'F+P'),
        'petri17': ('S44-G', 45, 'F+P'),
        'petri18': ('S44-D', 0,  'F+P'),
        'petri19': ('S46-G', 45, 'F+P'),
        'petri20': ('S46-D', 0,  'F+P'),
    },
     'batch#3': {
         'petri21': ('S33-G', 45, 'MNT'),
         'petri22': ('S33-D', 0,  'MNT'),
         'petri23': ('S37-G', 45, 'MNT'),
         'petri24': ('S37-D', 0,  'MNT'),
         'petri25': ('S30-G', 45, 'MNT'),
         'petri26': ('S30-D', 0,  'MNT'),
         'petri27': ('S32-G', 45, 'M+P'),
         'petri28': ('S32-D', 0,  'M+P'),
         'petri29': ('S36-G', 45, 'M+P'),
         'petri30': ('S36-D', 0,  'M+P'),
         'petri31': ('S27-G', 45, 'M+P'),
         'petri32': ('S27-D', 0,  'M+P'),
     },
    'batch#4': {
         'petri33': ('S29-G', 0,  'MNT'),
         'petri34': ('S29-D', 0,  'MNT'),
         'petri35': ('S31-G', 45, 'MNT'),
         'petri36': ('S31-D', 0,  'MNT'),
         'petri37': ('S34-G', 45, 'M+P'),
         'petri38': ('S34-D', 0,  'M+P'),

     },
}

MOYENNE = False   # True -> une valeur moyennée par pétri (lecteur_données_moy)
                  # False -> 3 spectres par pétri, un par zone (z1/z2/z3)

N_MAX_COMPOSANTES = 15  # borne supérieure explorée par le test de sélection


# ────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ────────────────────────────────────────────────────────────────────────────
def charger_spectres(config, etat, moyenne=False):
    """Charge tous les spectres et construit les étiquettes associées.

    Si moyenne=False, chaque échantillon donne 3 spectres (z1, z2, z3) qui
    partagent le même identifiant de souris -> ils resteront groupés lors
    de la validation croisée (LeaveOneGroupOut par souris).
    """
    spectres, etiquettes = [], []
    w = None

    for batch, petris in config.items():
        for petri, (echantillon, dose, type_) in petris.items():
            if moyenne:
                a_lire = [(None, lecteur_données_moy(batch, petri))]
            elif etat == 'frais':
                a_lire = [(z, lecteur_données_frais(batch, petri, z)) for z in ['z1', 'z2', 'z3']]
                etat = 'frais'
            else:
                a_lire = [(z, lecteur_données_fixes(batch, petri, z)) for z in ['z1', 'z2', 'z3']]
                etat = 'fixe'

            for zone, liste_fichiers in a_lire:
                if not liste_fichiers:
                    continue

                w_local, i = traiter_acquisitions_gellose(liste_fichiers)
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
    """Extrait échantillon, dose, sexe, traitement, id souris, état (frais/fixe)."""
    echantillons, doses, sexes, traitements, souris_id, etats = [], [], [], [], [], []

    for e in etiquettes:
        parts = e.split('_')
        echantillon = parts[0]
        reste = parts[-2]   # ex: "45FNT" (l'avant-dernier segment, avant l'état)

        m = re.match(r'(\d+)([A-Z])(.*)', reste)
        dose, sexe, traitement = int(m.group(1)), m.group(2), m.group(3)
        mouse = echantillon.split('-')[0]

        echantillons.append(echantillon)
        doses.append(dose)
        sexes.append(sexe)
        traitements.append(traitement)
        souris_id.append(mouse)
        etats.append(parts[-1])  # "frais" ou "fixe"

    return (
        np.array(echantillons), np.array(doses),
        np.array(sexes), np.array(traitements),
        np.array(souris_id), np.array(etats),
    )



def choisir_n_composantes(X, y, groupes, n_max=8, titre="Choix du nombre de composantes"):
    """Balaie le nombre de composantes PCA et évalue la balanced accuracy en
    validation croisée LeaveOneGroupOut, pour aider à choisir combien en
    garder avant le LDA final.
    Ce que fait la fonction
Objectif : trouver combien de composantes PCA garder avant d'entraîner un LDA final, en testant plusieurs valeurs et en comparant leurs performances.
Étape par étape :

valeurs_n = list(range(1, n_max + 1)) → elle va tester 1 composante, puis 2, puis 3... jusqu'à n_max (19 par défaut).
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

# ════════════════════════════════════════════════════════════════════════════
# Analyse dose (0gy vs 45gy)
# ════════════════════════════════════════════════════════════════════════════
def analyser_dose(X, doses, souris_id, masque, titre_suffixe, n_max=N_MAX_COMPOSANTES):
    X_sub = X[masque]
    y_sub = np.array([f"{d}gy" for d in doses[masque]])
    groupes_sub = souris_id[masque]

    n_pca = choisir_n_composantes(X_sub, y_sub, groupes_sub, n_max,
                                   f"Choix N_PCA — {titre_suffixe}")

    y_pred, ba = evaluer_lda(X_sub, y_sub, groupes_sub, n_pca, f"DOSE — {titre_suffixe}")

    # La PCA/LDA "finales" (fit sur tout X_sub) servent uniquement à reconstruire
    # le spectre discriminant LD1 — pas à évaluer la performance (ba vient de la CV ci-dessus)
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_sub)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_pca, y_sub)   # (n_échantillons, 1) — 2 classes

    return y_sub, y_pred, ba, n_pca, X_lda[:, 0], pca, lda

def analyser_traitement(X, traitements, souris_id, masque, titre_suffixe):
    X_sub = X[masque]
    y_sub = traitements[masque]          # ← on compare NT vs +P, pas 0gy vs 45gy
    groupes_sub = souris_id[masque]

    n_pca = choisir_n_composantes(X_sub, y_sub, groupes_sub, N_MAX_COMPOSANTES,
                                   f"Choix N_PCA — {titre_suffixe}")

    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_sub)
    y_pred, ba = evaluer_lda(X_pca, y_sub, groupes_sub, f"TRAITEMENT — {titre_suffixe}")

    return y_sub, y_pred, ba, n_pca



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
            ha='center', fontsize=8, color=couleur,
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


# ── Chargement des deux états ──────────────────────────────────────────────
X_frais, etiquettes_frais, w_frais = charger_spectres(CONFIG, 'frais', moyenne=MOYENNE)
X_fixe, etiquettes_fixe, w_fixe = charger_spectres(CONFIG, 'fixe', moyenne=MOYENNE)

X = np.concatenate([X_frais, X_fixe], axis=0)
etiquettes = etiquettes_frais + etiquettes_fixe   # ce sont des listes Python, "+" les concatène
w = w_frais   # en supposant que w_frais == w_fixe (mêmes wavenumbers pour les deux états)

echantillons, doses, sexes, traitements, souris_id, etats = parser_etiquettes(etiquettes)

# ════════════════════════════════════════════════════════════════════════════
# Analyse dose (0gy vs 45gy) - NT vs +P - sexe confondu et frais
# ════════════════════════════════════════════════════════════════════════════

#masque_NT = (etats == 'frais') & (traitements == 'NT')
#masque_P = (etats == 'frais') & (traitements == '+P')


#y_NT, y_pred_NT, ba_NT, n_pca_NT, ld1_NT, pca_NT, lda_NT = analyser_dose(
#    X, doses, souris_id, masque_NT, "NT - Frais - 0 Gy vs 45 Gy"
#)
#y_P, y_pred_P, ba_P, n_pca_P, ld1_P, pca_P, lda_P = analyser_dose(
#    X, doses, souris_id, masque_P, "+P - Frais - 0 Gy vs 45 Gy"
#)


#from matplotlib.gridspec import GridSpec

#fig = plt.figure(figsize=(11, 9))
#gs = GridSpec(2, 2, height_ratios=[1, 1.1], figure=fig)

#ax_cm_NT = fig.add_subplot(gs[0, 0])
#ax_cm_P = fig.add_subplot(gs[0, 1])
#ax_ld1 = fig.add_subplot(gs[1, :])   # occupe toute la largeur, en bas

#ConfusionMatrixDisplay.from_predictions(y_NT, y_pred_NT, ax=ax_cm_NT, colorbar=False, normalize='true', im_kw={'vmin': 0, 'vmax': 1})
#ax_cm_NT.set_title(f"NT ({n_pca_NT} comp., BA={ba_NT:.1%})")

#ConfusionMatrixDisplay.from_predictions(y_P, y_pred_P, ax=ax_cm_P, colorbar=False, normalize='true', im_kw={'vmin': 0, 'vmax': 1})
#ax_cm_P.set_title(f"+P({n_pca_P} comp., BA={ba_P:.1%})")

# ── Spectres discriminants LD1 (fixe vs frais), superposés ────────────────
#disc_NT = pca_NT.components_.T @ lda_NT.scalings_[:, 0]
#disc_P = pca_P.components_.T @ lda_P.scalings_[:, 0]

# Normalisation (norme unitaire) pour rendre les deux spectres comparables
# disc_NT = disc_NT / np.linalg.norm(disc_NT)
#disc_P = disc_P / np.linalg.norm(disc_P)

#ax_ld1.plot(w, disc_NT, label='NT', color='tab:orange', lw=1.2)
#ax_ld1.plot(w, disc_P, label='+P', color='tab:green', lw=1.2)

#marquer_positions(ax_ld1, w, disc_NT, [1094], couleur='tab:orange')
#marquer_positions(ax_ld1, w, disc_P, [1094], couleur='tab:green')
#marquer_positions(ax_ld1, w, disc_NT, [1244], couleur='tab:orange')
#marquer_positions(ax_ld1, w, disc_P, [1244], couleur='tab:green')

#ax_ld1.axhline(0, color='grey', lw=0.5)
#ax_ld1.set_xlabel("Nombre d'onde (cm⁻¹)")
#ax_ld1.set_ylabel("Poids LD1 (normalisé)")
#ax_ld1.set_title("Spectre discriminant LD1 — effet de la dose femelle vs mâle — NT et frais  ")
#annoter_pics(ax_ld1, w, disc_NT, n_pics=40)
#annoter_pics(ax_ld1, w, disc_P, n_pics=40)
#ax_ld1.legend(fontsize=8)

#plt.tight_layout()
#plt.show()

# ════════════════════════════════════════════════════════════════════════════
# Analyse dose (0gy vs 45gy) au sein de NT, séparément pour frais et fixe
# ════════════════════════════════════════════════════════════════════════════
#masque_fixe = (etats == 'fixe') & (traitements == 'NT')
#masque_frais = (etats == 'frais') & (traitements == 'NT')

#y_fixe, y_pred_fixe, ba_fixe, n_pca_fixe, ld1_fixe, pca_fixe, lda_fixe = analyser_dose(
#    X, doses, souris_id, masque_fixe, "Dose — Fixe"
#)
#y_frais, y_pred_frais, ba_frais, n_pca_frais, ld1_frais, pca_frais, lda_frais = analyser_dose(
#    X, doses, souris_id, masque_frais, "Dose — Frais"
#)

#from matplotlib.gridspec import GridSpec

#fig = plt.figure(figsize=(11, 9))
#gs = GridSpec(2, 2, height_ratios=[1, 1.1], figure=fig)

#ax_cm_fixe = fig.add_subplot(gs[0, 0])
#ax_cm_frais = fig.add_subplot(gs[0, 1])
#ax_ld1 = fig.add_subplot(gs[1, :])   # occupe toute la largeur, en bas

#ConfusionMatrixDisplay.from_predictions(y_fixe, y_pred_fixe, ax=ax_cm_fixe, colorbar=False, normalize='true', im_kw={'vmin': 0, 'vmax': 1})
#ax_cm_fixe.set_title(f"Fixe ({n_pca_fixe} comp., BA={ba_fixe:.1%})")

#ConfusionMatrixDisplay.from_predictions(y_frais, y_pred_frais, ax=ax_cm_frais, colorbar=False, normalize='true', im_kw={'vmin': 0, 'vmax': 1})
#ax_cm_frais.set_title(f"Frais ({n_pca_frais} comp., BA={ba_frais:.1%})")

# ── Spectres discriminants LD1 (fixe vs frais), superposés ────────────────
#disc_fixe = pca_fixe.components_.T @ lda_fixe.scalings_[:, 0]
#disc_frais = pca_frais.components_.T @ lda_frais.scalings_[:, 0]

# Normalisation (norme unitaire) pour rendre les deux spectres comparables
#disc_fixe = disc_fixe / np.linalg.norm(disc_fixe)
#disc_frais = disc_frais / np.linalg.norm(disc_frais)

#ax_ld1.plot(w, disc_fixe, label='Fixe', color='tab:orange', lw=1.2)
#ax_ld1.plot(w, disc_frais, label='Frais', color='tab:green', lw=1.2)

#ax_ld1.axhline(0, color='grey', lw=0.5)
#ax_ld1.set_xlabel("Nombre d'onde (cm⁻¹)")
#ax_ld1.set_ylabel("Poids LD1 (normalisé)")
#ax_ld1.set_title("Spectre discriminant LD1 — Dose, Fixe vs Frais")
#ax_ld1.legend(fontsize=8)

#plt.tight_layout()
#plt.show()

# ════════════════════════════════════════════════════════════════════════════
# Analyse dose (0gy vs 45gy) - NT vs +P - sexe confondu et frais
# ════════════════════════════════════════════════════════════════════════════

masque_NT = (etats == 'frais') & (traitements == 'NT')

y_NT, y_pred_NT, ba_NT, n_pca_NT, ld1_NT, pca_NT, lda_NT = analyser_dose(
    X, doses, souris_id, masque_NT, "NT - Frais - 0 Gy vs 45 Gy"
)

# ════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Matrice de confusion (avec colorbar)
# ════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_NT, y_pred_NT, ax=ax1,
    colorbar=True,              # ← gradient affiché
    normalize='true',
    im_kw={'vmin': 0, 'vmax': 1}
)
ax1.set_title(f"NT ({n_pca_NT} comp., BA={ba_NT:.1%})")

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Spectre discriminant LD1
# ════════════════════════════════════════════════════════════════════════
disc_NT = pca_NT.components_.T @ lda_NT.scalings_[:, 0]
disc_NT = disc_NT / np.linalg.norm(disc_NT)   # normalisation (optionnel mais cohérent avec vos autres figures)

fig2, ax2 = plt.subplots(figsize=(11, 6))

ax2.plot(w, disc_NT, label='NT', color='tab:orange', lw=1.2)
ax2.axhline(0, color='grey', lw=0.5)
ax2.set_xlabel("Nombre d'onde (cm⁻¹)")
ax2.set_ylabel("Poids LD1 (normalisé)")
ax2.set_title("Spectre discriminant LD1 — NT, frais — 0 Gy vs 45 Gy")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()



