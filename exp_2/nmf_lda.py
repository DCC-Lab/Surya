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


from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, balanced_accuracy_score, ConfusionMatrixDisplay

from extract_data import traiter_acquisitions_gellose, lecteur_données_frais, lecteur_données_fixes, lecteur_données_moy, soustraire_spectre, lecteur_gelose






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
    #'batch#4': {
    #     'petri33': ('S29-G', 0,  'MNT'),
    #     'petri34': ('S29-D', 0,  'MNT'),
    #     'petri35': ('S31-G', 45, 'MNT'),
    #     'petri36': ('S31-D', 0,  'MNT'),
    #     'petri37': ('S34-G', 45, 'M+P'),
    #     'petri38': ('S34-D', 0,  'M+P'),

    # },
}

MOYENNE = False   # True -> une valeur moyennée par pétri (lecteur_données_moy)
                  # False -> 3 spectres par pétri, un par zone (z1/z2/z3)

N_MAX_COMPOSANTES = 11  # borne supérieure explorée par le test de sélection



# ────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ────────────────────────────────────────────────────────────────────────────
def charger_spectres(config, etat, i_nocif, moyenne=False):
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

                i_corr = soustraire_spectre(w_local, i, w_local, i_nocif)

                w = w_local
                suffixe = f"_{zone}" if zone else ""
                spectres.append(i_corr)
                etiquettes.append(f"{echantillon}{suffixe}_{dose}{type_}_{etat}")

    return np.array(spectres), etiquettes, w


def charger_nocif(config):
    i_s = []

    for batch, petris in config.items():
        for petri, (echantillon, dose, type_) in petris.items():
            fichiers = lecteur_gelose(batch, petri)
            if not fichiers:
                continue
            w, i = traiter_acquisitions_gellose(fichiers)
            i_s.append(i)
            if not i_s:
                raise ValueError("Aucun spectre de gélose (nocif) n'a pu être chargé.")
            i_arr = np.array(i_s)
    return np.mean(i_arr, axis=0)

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
    """Balaie le nombre de composantes nmf et évalue la balanced accuracy en
    validation croisée LeaveOneGroupOut, pour aider à choisir combien en
    garder avant le LDA final.
    Ce que fait la fonction
Objectif : trouver combien de composantes nmf garder avant d'entraîner un LDA final, en testant plusieurs valeurs et en comparant leurs performances.
Étape par étape :

valeurs_n = list(range(1, n_max + 1)) → elle va tester 1 composante, puis 2, puis 3... jusqu'à n_max (19 par défaut).
Pour chaque valeur n de composantes :

elle construit un pipeline : d'abord une nmf qui réduit X à n dimensions, puis un LDA (Linear Discriminant Analysis) qui classe les données à partir de ces n dimensions.
elle évalue ce pipeline avec cross_val_predict et une validation croisée LeaveOneGroupOut (LOGO) : à chaque tour, un groupe entier (par exemple un sujet, une session...) est mis de côté comme test, et le modèle est entraîné sur tous les autres groupes. Ça évite les fuites de données si plusieurs échantillons viennent de la même source.
elle calcule le balanced accuracy score entre les vraies étiquettes y et les prédictions, et stocke ce score.


Elle trace un graphique : score en fonction du nombre de composantes, avec une ligne horizontale à 0.5 (niveau du hasard pour un problème à 2 classes).
Elle retourne le n qui donne le meilleur score, et l'affiche.

En résumé : c'est une recherche du nombre optimal de composantes nmf, en observant à partir de combien de composantes l'ajout de dimensions supplémentaires n'améliore plus (ou dégrade) la performance de classification.
"""
    #n_max = min(n_max, X.shape[0] - 1, X.shape[1])
    valeurs_n = list(range(1, n_max + 1))
    scores = []
    logo = LeaveOneGroupOut()

    X_nn = X - X.min()

    for n in valeurs_n:
        pipe = Pipeline([
            ('nmf', NMF(n_components=n)),
            ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')),
        ])
        y_pred = cross_val_predict(pipe, X_nn, y, groups=groupes, cv=logo)
        scores.append(balanced_accuracy_score(y, y_pred))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valeurs_n, scores, marker='o')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0.5, color='grey', lw=0.5, linestyle='--', label='hasard (2 classes)')
    ax.set_xlabel("Nombre de composantes nmf")
    ax.set_ylabel("Balanced accuracy (CV LeaveOneGroupOut)")
    ax.set_title(titre)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    meilleur_n = valeurs_n[int(np.argmax(scores))]
    print(f"{titre} — meilleur score : {max(scores):.1%} avec {meilleur_n} composante(s)\n")
    return meilleur_n




def evaluer_lda(X_sub, y, groupes, n_nmf, titre):
    """Pipeline nmf+LDA, entièrement re-fit à chaque pli LeaveOneGroupOut."""
    pipe = Pipeline([
        ('nmf', NMF(n_components=n_nmf)),
        ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')),
    ])

    X_nmf = X_sub - X_sub.min()   
    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(pipe, X_nmf, y, groups=groupes, cv=logo)

    ba = balanced_accuracy_score(y, y_pred)
    print(f"── {titre} ──")
    print(classification_report(y, y_pred))
    print(f"Balanced accuracy : {ba:.1%}\n")

    return y_pred, ba

# ════════════════════════════════════════════════════════════════════════════
# Analyse dose (0gy vs 45gy)
# ════════════════════════════════════════════════════════════════════════════


def entrainer_lda(X, y_labels, souris_id, masque, titre_suffixe, classe_cible,
                   n_max=N_MAX_COMPOSANTES):
    """
    Entraîne un LDA (2 classes) discriminant sur les échantillons sélectionnés
    par `masque`, avec y_labels comme étiquette de classe.

    classe_cible : la classe qui doit correspondre au signe positif de l'axe
                   LD1 et du spectre de charge (ex: "45gy" ou "+P").
    """
    X_sub = X[masque]
    y_sub = y_labels[masque]
    groupes_sub = souris_id[masque]

    n_nmf = choisir_n_composantes(X_sub, y_sub, groupes_sub, n_max,
                                   f"Choix N_NMF — {titre_suffixe}")

    y_pred, ba = evaluer_lda(X_sub, y_sub, groupes_sub, n_nmf, titre_suffixe)

    # nmf/LDA "finales" fit sur tout X_sub — servent à reconstruire le LD1
    X_nn = X_sub - X_sub.min()
    nmf = NMF(n_components=n_nmf)
    X_nmf = nmf.fit_transform(X_nn)
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    X_lda = lda.fit_transform(X_nmf, y_sub)
    ld1 = X_lda[:, 0]

    # ── Convention de signe ──────────────────────────────────────────
    # On force : score LD1 moyen le plus élevé = classe_cible
    score_cible = ld1[y_sub == classe_cible].mean()
    score_autre = ld1[y_sub != classe_cible].mean()
    signe = 1 if score_cible > score_autre else -1

    ld1 = signe * ld1
    loading_spec = signe * (lda.coef_[0] @ nmf.components_)  # spectre de charge signé

    print(f"{titre_suffixe} : positif = '{classe_cible}' "
          f"(score moyen {signe*score_cible:.3f} vs {signe*score_autre:.3f})")

    return y_sub, y_pred, ba, n_nmf, ld1, nmf, lda, loading_spec



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
X_frais, etiquettes_frais, w_frais = charger_spectres(CONFIG, 'frais', charger_nocif(CONFIG), moyenne=MOYENNE)
X_fixe, etiquettes_fixe, w_fixe = charger_spectres(CONFIG, 'fixe', charger_nocif(CONFIG), moyenne=MOYENNE)

X = np.concatenate([X_frais, X_fixe], axis=0)
etiquettes = etiquettes_frais + etiquettes_fixe   # ce sont des listes Python, "+" les concatène
w = w_frais   # en supposant que w_frais == w_fixe (mêmes wavenumbers pour les deux états)

echantillons, doses, sexes, traitements, souris_id, etats, zones = parser_etiquettes(etiquettes)




# étiquette de dose sous forme de chaîne, pour tout le dataset
y_dose = np.array([f"{d}gy" for d in doses])



masque_NTFi = (etats == 'frais') & (traitements == 'NT') & (sexes == 'F')
masque_NTFr = (etats == 'frais') & (doses == 45) & (sexes == 'F')


y_NTFi, y_pred_NTFi, ba_NTFi, n_nmf_NTFi, ld1_NTFi, nmf_NTFi, lda_NTFi, loading_NTFi = entrainer_lda(
    X, y_dose, souris_id, masque_NTFi, "NT - Frais - 0 Gy vs 45 Gy",
    classe_cible="45gy"     # ← explicite : positif = 45 Gy
)

y_NTFr, y_pred_NTFr, ba_NTFr, n_nmf_NTFr, ld1_NTFr, nmf_NTFr, lda_NTFr, loading_NTFr = entrainer_lda(
    X, traitements, souris_id, masque_NTFr, "45 - Frais - +P vs NT",
    classe_cible="+P"       # ← explicite : positif = +P (pansement)
)




# ════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Matrice de confusion (avec colorbar)
# ════════════════════════════════════════════════════════════════════════
#fig1, ax1 = plt.subplots(figsize=(6, 5))

#ConfusionMatrixDisplay.from_predictions(
#    y_NTFi, y_pred_NTFi, ax=ax1,
#    colorbar=True,              # ← gradient affiché
#    normalize='true',
#    im_kw={'vmin': 0, 'vmax': 1}, 
#    cmap='RdPu',
#    display_labels=["Non-irradiated", "Irradiated"]
#)
#ax1.set_title(f"Effect of irradiation - female fresh ({n_nmf_NTFi} comp., BA={ba_NTFi:.1%})")

#plt.tight_layout()
#plt.show()


def score_ld1(spectres, nmf, lda):
    """Projette des spectres bruts sur un LD1 déjà figé (nmf+lda entraînés sur NT).
    Le signe indique la classe prédite, le seuil de décision est à 0."""
    X_nmf = nmf.transform(spectres)
    return lda.decision_function(X_nmf)




# ════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Matrice de confusion (avec colorbar)
# ════════════════════════════════════════════════════════════════════════
#fig1, ax3 = plt.subplots(figsize=(6, 5))

#ConfusionMatrixDisplay.from_predictions(
#    y_NTFr, y_pred_NTFr, ax=ax3,
#    labels=['+P', 'NT'],
#    colorbar=True,              # ← gradient affiché
#    normalize='true',
#    im_kw={'vmin': 0, 'vmax': 1}, 
#    cmap='RdPu',
#    display_labels=["Pansement", "Non traité"]
#)
#ax3.set_title(f"Effect of pansement - fresh ({n_nmf_NTFr} comp., BA={ba_NTFr:.1%})")

#plt.tight_layout()
#plt.show()






# ════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Spectre discriminant LD1
# ════════════════════════════════════════════════════════════════════════
disc_NTFi = loading_NTFi / np.linalg.norm(loading_NTFi)
disc_NTFr = loading_NTFr / np.linalg.norm(loading_NTFr)


def bootstrap_loading(X_sub, y_sub, groupes_sub, n_nmf, classe_cible, n_boot=30):
    loadings = []
    n = len(y_sub)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_b, y_b = X_sub[idx] - X_sub[idx].min(), y_sub[idx]
        if len(np.unique(y_b)) < 2:
            continue
        nmf_b = NMF(n_components=n_nmf).fit(X_b)
        X_nmf_b = nmf_b.transform(X_b)
        lda_b = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto').fit(X_nmf_b, y_b)
        ld1_b = lda_b.transform(X_nmf_b)[:, 0]

        # même convention de signe que dans entrainer_lda
        score_cible = ld1_b[y_b == classe_cible].mean()
        score_autre = ld1_b[y_b != classe_cible].mean()
        signe = 1 if score_cible > score_autre else -1

        loading_b = signe * (lda_b.coef_[0] @ nmf_b.components_)
        loadings.append(loading_b / np.linalg.norm(loading_b))
    return np.array(loadings)

boots = bootstrap_loading(X[masque_NTFi], y_dose[masque_NTFi], souris_id[masque_NTFi],
                           n_nmf_NTFi, classe_cible="45gy")

fig, ax = plt.subplots(figsize=(11,5))
for b in boots:
    ax.plot(w, b, color='grey', alpha=0.15)
ax.plot(w, disc_NTFi, color='xkcd:scarlet', lw=1.5, label='Fit original')
ax.axhline(0, color='k', lw=0.5)
ax.legend()
plt.show()

fig2, ax2 = plt.subplots(figsize=(11, 6))
ax2.plot(w, disc_NTFi, label='Effet dose', color='xkcd:scarlet', lw=1.2)
ax2.plot(w, disc_NTFr, label='Effet pansement', color='tab:green', lw=1.2)
ax2.axhline(0, color='grey', lw=0.5)
ax2.set_xlabel("Raman shift(cm⁻¹)")
ax2.set_ylabel("LD1 weight")
ax2.set_title("Effet de l'irradiation vs effet du pansement")
#annoter_pics(ax2, w, disc_NTFi, n_pics=50, couleur='black')
#annoter_pics(ax2, w, disc_NTFr, n_pics=50, couleur='black')
ax2.legend()
plt.tight_layout()
plt.show()
