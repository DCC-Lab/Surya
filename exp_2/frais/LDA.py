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

from extract_data import traiter_acquisitions_gellose, lecteur_données_fixes, lecteur_données_frais, lecteur_données_moy


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
    # 'batch#3': {
    #     'petri21': ('S33-G', 45, 'MNT'),
    #     'petri22': ('S33-D', 0,  'MNT'),
    #     'petri23': ('S37-G', 45, 'MNT'),
    #     'petri24': ('S37-D', 0,  'MNT'),
    #     'petri25': ('S30-G', 45, 'MNT'),
    #     'petri26': ('S30-D', 0,  'MNT'),
    #     'petri27': ('S32-G', 45, 'M+P'),
    #     'petri28': ('S32-D', 0,  'M+P'),
    #     'petri29': ('S36-G', 45, 'M+P'),
    #     'petri30': ('S36-D', 0,  'M+P'),
    #     'petri31': ('S27-G', 45, 'M+P'),
    #     'petri32': ('S27-D', 0,  'M+P'),
    # },
}

MOYENNE = False   # True -> une valeur moyennée par pétri (lecteur_données_moy)
                  # False -> 3 spectres par pétri, un par zone (z1/z2/z3)

N_MAX_COMPOSANTES = 17  # borne supérieure explorée par le test de sélection


# ────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ────────────────────────────────────────────────────────────────────────────
def charger_spectres(config, moyenne=False):
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
            else:
                a_lire = [(z, lecteur_données_frais(batch, petri, z)) for z in ['z1', 'z2', 'z3']]

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
                etiquettes.append(f"{echantillon}{suffixe}_{dose}{type_}")

    return np.array(spectres), etiquettes, w


def parser_etiquettes(etiquettes):
    """Extrait échantillon, dose, sexe, traitement, id souris depuis les étiquettes.

    Gère les deux formats possibles :
      - "S48-G_z1_45FNT"  (3 segments, moyenne=False)
      - "S48-G_45FNT"     (2 segments, moyenne=True)
    """
    echantillons, doses, sexes, traitements, souris_id = [], [], [], [], []

    for e in etiquettes:
        parts = e.split('_')
        echantillon = parts[0]
        reste = parts[-1]

        m = re.match(r'(\d+)([A-Z])(.*)', reste)
        dose, sexe, traitement = int(m.group(1)), m.group(2), m.group(3)
        mouse = echantillon.split('-')[0]   # "S48-G" -> "S48"

        echantillons.append(echantillon)
        doses.append(dose)
        sexes.append(sexe)
        traitements.append(traitement)
        souris_id.append(mouse)

    return (
        np.array(echantillons), np.array(doses),
        np.array(sexes), np.array(traitements), np.array(souris_id),
    )


# ────────────────────────────────────────────────────────────────────────────
# OUTILS
# ────────────────────────────────────────────────────────────────────────────
def choisir_n_composantes(X, y, groupes, n_max=19, titre="Choix du nombre de composantes"):
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

    #fig, ax = plt.subplots(figsize=(8, 5))
    #ax.plot(valeurs_n, scores, marker='o')
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.axhline(0.5, color='grey', lw=0.5, linestyle='--', label='hasard (2 classes)')
    #ax.set_xlabel("Nombre de composantes PCA")
    #ax.set_ylabel("Balanced accuracy (CV LeaveOneGroupOut)")
    #ax.set_title(titre)
    #ax.legend(fontsize=8)
    #plt.tight_layout()
    #plt.show()

    meilleur_n = valeurs_n[int(np.argmax(scores))]
    print(f"{titre} — meilleur score : {max(scores):.1%} avec {meilleur_n} composante(s)\n")
    return meilleur_n


def evaluer_lda(X_pca, y, groupes, titre):
    """LDA + validation croisée LeaveOneGroupOut (par souris) + rapport de classification."""
    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, groups=groupes, cv=logo)

    ba = balanced_accuracy_score(y, y_pred)
    print(f"── {titre} ──")
    print(classification_report(y, y_pred))
    print(f"Balanced accuracy : {ba:.1%}\n")

    return y_pred, ba




# (décommentez 'batch#2' dans CONFIG en haut du fichier)

def analyser_dose_dans_sous_groupe(X, groupes_dose, souris_id, masque, titre_suffixe, couleur):
    """Lance le pipeline complet (choix n_pca, LDA, rapport, matrice de confusion,
    spectre discriminant) sur un sous-ensemble des données (ex: seulement NT, ou +P)."""
    X_sub = X[masque]
    y_sub = groupes_dose[masque]
    groupes_sub = souris_id[masque]

    n_pca = choisir_n_composantes(X_sub, y_sub, groupes_sub, N_MAX_COMPOSANTES,
                                   f"Choix N_PCA — {titre_suffixe}")

    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_sub)
    y_pred, ba = evaluer_lda(X_pca, y_sub, groupes_sub, f"DOSE — {titre_suffixe}")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y_sub)

    discriminant = pca.components_.T @ lda.scalings_[:, 0]

    # ── Figure combinée : matrice de confusion + spectre discriminant ────────
    #fig, (ax_cm, ax_spec) = plt.subplots(1, 2, figsize=(14, 4.5))

    #ConfusionMatrixDisplay.from_predictions(
    #    y_sub, y_pred, ax=ax_cm, colorbar=False, normalize='true'
    #)
    #ax_cm.set_title(f"Matrice de confusion — {titre_suffixe}\n({n_pca} composantes, BA={ba:.1%})")

    #ax_spec.plot(w, discriminant, color=couleur)
    #ax_spec.axhline(0, color='grey', lw=0.5)
    #ax_spec.set_xlabel("Raman shift (cm$^{-1}$)")
    #ax_spec.set_ylabel("Poids LD1")
    #ax_spec.set_title(f"Spectre discriminant — {titre_suffixe}")

    #plt.tight_layout()
    #plt.show()

    return n_pca, y_pred, ba, discriminant






# ════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
X, etiquettes, w = charger_spectres(CONFIG, moyenne=MOYENNE)
echantillons, doses, sexes, traitements, souris_id = parser_etiquettes(etiquettes)

groupes_dose = np.array([f"{d}gy" for d in doses])

# ════════════════════════════════════════════════════════════════════════════
# NOMBRE DE SOURIS PAR GROUPE
# ════════════════════════════════════════════════════════════════════════════

for t in np.unique(traitements):
    m_t = traitements == t
    souris_0 = set(str(s) for s in souris_id[m_t & (doses == 0)])
    souris_45 = set(str(s) for s in souris_id[m_t & (doses == 45)])
    communes = souris_0 & souris_45

    print(f"── {t} ──")
    print(f"  {len(souris_0)} souris à 0gy  : {sorted(souris_0)}")
    print(f"  {len(souris_45)} souris à 45gy : {sorted(souris_45)}")
    print(f"  souris présentes aux deux doses : {len(communes)} → {sorted(communes)}")
    print()

# ════════════════════════════════════════════════════════════════════════════
# 1) DOSE, à l'intérieur du groupe NT seulement (0gy_NT vs 45gy_NT)
# ════════════════════════════════════════════════════════════════════════════
masque_nt = traitements == 'NT'
n_pca_nt, y_pred_nt, ba_nt, disc_nt = analyser_dose_dans_sous_groupe(
    X, groupes_dose, souris_id, masque_nt, "NT seul", 'darkblue'
)

# ════════════════════════════════════════════════════════════════════════════
# 2) DOSE, à l'intérieur du groupe +P seulement (0gy_+P vs 45gy_+P)
# ════════════════════════════════════════════════════════════════════════════
#masque_p = traitements == '+P'
#n_pca_p, y_pred_p, ba_p, disc_p = analyser_dose_dans_sous_groupe(
#    X, groupes_dose, souris_id, masque_p, "+P seul", 'darkgreen'
#)

# ════════════════════════════════════════════════════════════════════════════
# 3) DOSE globale, peu importe le traitement (0gy vs 45gy, tous confondus)
# ════════════════════════════════════════════════════════════════════════════
#n_pca_dose, y_pred_dose, ba_dose, disc_dose = analyser_dose_dans_sous_groupe(
#    X, groupes_dose, souris_id, np.ones(len(X), dtype=bool), "toutes conditions", 'darkred'
#)



from scipy.signal import find_peaks

def annoter_pics(ax, w, spectre, n_pics=40, couleur='black'):
    """Détecte les n_pics plus grands pics (en valeur absolue) et les annote
    avec leur position en cm-1."""
    idx_pos, _ = find_peaks(spectre)
    idx_neg, _ = find_peaks(-spectre)
    idx_tous = np.concatenate([idx_pos, idx_neg])

    amplitudes = np.abs(spectre[idx_tous])
    ordre = np.argsort(amplitudes)[::-1][:n_pics]
    idx_principaux = idx_tous[ordre]

    for idx in idx_principaux:
        x, y = w[idx], spectre[idx]
        ax.annotate(
            f"{x:.0f}",
            xy=(x, y),
            xytext=(0, 10 if y >= 0 else -15),
            textcoords='offset points',
            ha='center', fontsize=8, color=couleur,
            arrowprops=dict(arrowstyle='-', lw=0.5, color=couleur),
        )


fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(w, disc_nt, color='darkblue')
ax.axhline(0, color='grey', lw=0.5)
ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("Poids LD1")
ax.set_title("Spectre discriminant — Dose (0gy vs 45gy), groupe NT seul")

annoter_pics(ax, w, disc_nt, n_pics=40, couleur='darkblue')

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# Qu'est-ce qui est sain et qu'est-ce qui ne l'ai pas
# ════════════════════════════════════════════════════════════════════════════
# ── Étape 1 : calibrer l'axe de référence UNIQUEMENT sur NT ──────────────────
#masque_nt = traitements == 'NT'
#X_nt = X[masque_nt]
#y_nt = groupes_dose[masque_nt]
#groupes_nt = souris_id[masque_nt]

#n_pca_nt = choisir_n_composantes(X_nt, y_nt, groupes_nt, N_MAX_COMPOSANTES,
#                                  "Choix N_PCA — NT (axe de référence)")

#pca_nt = PCA(n_components=n_pca_nt)
#X_pca_nt = pca_nt.fit_transform(X_nt)

#lda_nt = LinearDiscriminantAnalysis()
#lda_nt.fit(X_pca_nt, y_nt)

# ── Étape 2 : déterminer le sens de l'axe ─────────────────────────────────────
#scores_nt = lda_nt.transform(X_pca_nt)[:, 0]
#moyenne_0gy = scores_nt[y_nt == '0gy'].mean()
#moyenne_45gy = scores_nt[y_nt == '45gy'].mean()

#print(f"Score LD1 moyen — 0gy_NT (sain)     : {moyenne_0gy:.2f}")
#print(f"Score LD1 moyen — 45gy_NT (irradié) : {moyenne_45gy:.2f}")

#if moyenne_45gy > moyenne_0gy:
#    print("→ LD1 positif = vers l'irradiation, LD1 négatif = vers le sain")
#else:
#    print("→ LD1 positif = vers le sain, LD1 négatif = vers l'irradiation")

# ── Étape 3 : projeter TOUS les groupes (NT et +P) sur cet axe ────────────────
#X_pca_tous = pca_nt.transform(X)          # même PCA que celle entraînée sur NT
#scores_tous = lda_nt.transform(X_pca_tous)[:, 0]  # projection sur l'axe NT

#groupes_4 = np.array([f"{d}gy_{t}" for d, t in zip(doses, traitements)])

# ── Étape 4 : graphique ────────────────────────────────────────────────────────
#color_map = {'0gy_NT': 'tab:blue', '45gy_NT': 'tab:red',
#             '0gy_+P': 'tab:cyan', '45gy_+P': 'tab:orange'}

#fig, ax = plt.subplots(figsize=(10, 5))
#rng = np.random.default_rng(0)
#for g in ['0gy_NT', '45gy_NT', '0gy_+P', '45gy_+P']:
#    m = groupes_4 == g
#    y_jitter = rng.normal(0, 0.05, m.sum())  # juste pour espacer visuellement les points
#    ax.scatter(scores_tous[m], y_jitter, label=g, color=color_map[g],
#               s=60, edgecolors='k', linewidths=0.3)

#ax.axvline(moyenne_0gy, color='tab:blue', linestyle='--', lw=1)
#ax.axvline(moyenne_45gy, color='tab:red', linestyle='--', lw=1)
#ax.set_xlabel("Score sur l'axe LD1 (calibré sur NT seul : 0gy vs 45gy)")
#ax.set_yticks([])
#ax.set_title("Projection des 4 groupes sur l'axe « dommage radiatif » (référence NT)")
#ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
#plt.tight_layout()
#plt.show()

#spectre_moyen_0gy_nt = X[  (traitements=='NT') & (doses==0)].mean(axis=0)
#spectre_moyen_0gy_p  = X[  (traitements=='+P') & (doses==0)].mean(axis=0)
#diff_pansement = spectre_moyen_0gy_p - spectre_moyen_0gy_nt

#fig, ax = plt.subplots(figsize=(10,4))
#ax.plot(w, diff_pansement, color='purple')
#ax.axhline(0, color='grey', lw=0.5)
#ax.set_title("Effet du pansement seul (0gy_+P − 0gy_NT)")
#plt.show()

