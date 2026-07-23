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


# ── Chargement des deux états ──────────────────────────────────────────────
X_frais, etiquettes_frais, w_frais = charger_spectres(CONFIG, 'frais', moyenne=MOYENNE)
X_fixe, etiquettes_fixe, w_fixe = charger_spectres(CONFIG, 'fixe', moyenne=MOYENNE)

X = np.concatenate([X_frais, X_fixe], axis=0)
etiquettes = etiquettes_frais + etiquettes_fixe   # ce sont des listes Python, "+" les concatène
w = w_frais   # en supposant que w_frais == w_fixe (mêmes wavenumbers pour les deux états)

echantillons, doses, sexes, traitements, souris_id, etats = parser_etiquettes(etiquettes)

# ── Vérification rapide de la répartition ──────────────────────────────────
print("Répartition NT, par état :")
for e in np.unique(etats):
    for d in np.unique(doses):
        m = (traitements == '+P') & (etats == e) & (doses == d)
        if m.sum() > 0:
            print(f"  {e}, {d}gy : {m.sum()} spectres, {len(np.unique(souris_id[m]))} souris")

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
# Analyse dose (0gy vs 45gy) au sein de NT, séparément pour frais et fixe
# ════════════════════════════════════════════════════════════════════════════
def analyser_dose_par_etat(X, doses, souris_id, masque, titre_suffixe, n_max=N_MAX_COMPOSANTES):
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

def analyser_traitement_par_dose(X, traitements, souris_id, masque, titre_suffixe):
    X_sub = X[masque]
    y_sub = traitements[masque]          # ← on compare NT vs +P, pas 0gy vs 45gy
    groupes_sub = souris_id[masque]

    n_pca = choisir_n_composantes(X_sub, y_sub, groupes_sub, N_MAX_COMPOSANTES,
                                   f"Choix N_PCA — {titre_suffixe}")

    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_sub)
    y_pred, ba = evaluer_lda(X_pca, y_sub, groupes_sub, f"TRAITEMENT — {titre_suffixe}")

    return y_sub, y_pred, ba, n_pca



#masque_nt_frais = (traitements == '+P') & (etats == 'frais')
#masque_nt_fixe = (traitements == '+P') & (etats == 'fixe')

#y_nt_frais, y_pred_nt_frais, ba_nt_frais, n_pca_frais = analyser_dose_par_etat(
#    X, doses, souris_id, masque_nt_frais, "frais"
#)
#y_nt_fixe, y_pred_nt_fixe, ba_nt_fixe, n_pca_fixe = analyser_dose_par_etat(
#    X, doses, souris_id, masque_nt_fixe, "fixe"
#)



#masque_0gy = (doses == 0)
#masque_45gy = (doses == 45)

#y_0gy, y_pred_0gy, ba_0gy, n_pca_0gy = analyser_traitement_par_dose(
#    X, traitements, souris_id, masque_0gy, "0gy (NT vs +P)"
#)
#y_45gy, y_pred_45gy, ba_45gy, n_pca_45gy = analyser_traitement_par_dose(
#    X, traitements, souris_id, masque_45gy, "45gy (NT vs +P)"
#)

#print("Répartition NT — dose × état :")
#for e in np.unique(etats):
#    for d in np.unique(doses):
#        m = (traitements == '+P') & (etats == e) & (doses == d)
#        if m.sum() > 0:
#            souris = np.unique(souris_id[m])
#            print(f"  {e}, {d}gy : {m.sum()} spectres, {len(souris)} souris → {sorted(souris)}")


print("Répartition complète — traitement × dose × état :")
for t in np.unique(traitements):
    for e in np.unique(etats):
        for d in np.unique(doses):
            m = (traitements == t) & (etats == e) & (doses == d)
            if m.sum() > 0:
                souris = np.unique(souris_id[m])
                print(f"  {t}, {e}, {d}gy : {m.sum()} spectres, {len(souris)} souris → {sorted(souris)}")



#----------------------
# signature du dommage
#----------------------
# ── Construire LD1c : signature dommage, à partir de NT seul (0gy vs 45gy) ──
#masque_nt = (traitements == 'NT')
#X_nt = X[masque_nt]
#y_nt = np.array([f"{d}gy" for d in doses[masque_nt]])
#groupes_nt = souris_id[masque_nt]

#n_pca_nt = choisir_n_composantes(X_nt, y_nt, groupes_nt, N_MAX_COMPOSANTES,
#                                  "Choix N_PCA — NT (signature dommage)")

#pca_nt = PCA(n_components=n_pca_nt)
#X_pca_nt = pca_nt.fit_transform(X_nt)

#lda_nt = LinearDiscriminantAnalysis()
#lda_nt.fit(X_pca_nt, y_nt)

#disc_c = pca_nt.components_.T @ lda_nt.scalings_[:, 0]   # spectre discriminant brut (LD1c)

#def normaliser(v):
#    return v / np.linalg.norm(v)

#disc_c_norm = normaliser(disc_c)   # ← c'est cette variable qui manquait



#effet_dose_dans_NT = X[(doses==45)&(traitements=='NT')].mean(axis=0) - X[(doses==0)&(traitements=='NT')].mean(axis=0)
#effet_dose_dans_P  = X[(doses==45)&(traitements=='+P')].mean(axis=0) - X[(doses==0)&(traitements=='+P')].mean(axis=0)

#interaction = effet_dose_dans_P - effet_dose_dans_NT

masque_fixe = (etats == 'fixe') & (traitements == 'NT')
masque_frais = (etats == 'frais') & (traitements == 'NT')

y_fixe, y_pred_fixe, ba_fixe, n_pca_fixe, ld1_fixe, pca_fixe, lda_fixe = analyser_dose_par_etat(
    X, doses, souris_id, masque_fixe, "Dose — Fixe"
)
y_frais, y_pred_frais, ba_frais, n_pca_frais, ld1_frais, pca_frais, lda_frais = analyser_dose_par_etat(
    X, doses, souris_id, masque_frais, "Dose — Frais"
)


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(11, 9))
gs = GridSpec(2, 2, height_ratios=[1, 1.1], figure=fig)

ax_cm_fixe = fig.add_subplot(gs[0, 0])
ax_cm_frais = fig.add_subplot(gs[0, 1])
ax_ld1 = fig.add_subplot(gs[1, :])   # occupe toute la largeur, en bas

ConfusionMatrixDisplay.from_predictions(y_fixe, y_pred_fixe, ax=ax_cm_fixe, colorbar=False, normalize='true')
ax_cm_fixe.set_title(f"Fixe ({n_pca_fixe} comp., BA={ba_fixe:.1%})")

ConfusionMatrixDisplay.from_predictions(y_frais, y_pred_frais, ax=ax_cm_frais, colorbar=False, normalize='true')
ax_cm_frais.set_title(f"Frais ({n_pca_frais} comp., BA={ba_frais:.1%})")

# ── Spectres discriminants LD1 (fixe vs frais), superposés ────────────────
disc_fixe = pca_fixe.components_.T @ lda_fixe.scalings_[:, 0]
disc_frais = pca_frais.components_.T @ lda_frais.scalings_[:, 0]

# Normalisation (norme unitaire) pour rendre les deux spectres comparables
disc_fixe = disc_fixe / np.linalg.norm(disc_fixe)
disc_frais = disc_frais / np.linalg.norm(disc_frais)

ax_ld1.plot(w, disc_fixe, label='Fixe', color='tab:orange', lw=1.2)
ax_ld1.plot(w, disc_frais, label='Frais', color='tab:green', lw=1.2)

ax_ld1.axhline(0, color='grey', lw=0.5)
ax_ld1.set_xlabel("Nombre d'onde (cm⁻¹)")
ax_ld1.set_ylabel("Poids LD1 (normalisé)")
ax_ld1.set_title("Spectre discriminant LD1 — Dose, Fixe vs Frais")
ax_ld1.legend(fontsize=8)

plt.tight_layout()
plt.show()

