"""
Correction de l'effet d'étalonnage (etaloning) sur des spectres Raman.

L'etaloning produit une modulation quasi-sinusoïdale de l'efficacité
quantique du CCD, superposée au signal Raman. Ce script propose :

  1. detect_fringe_frequency() : identifie la fréquence dominante des
     franges via une FFT du spectre.
  2. remove_etalon_fft()       : filtre coupe-bande centré sur cette
     fréquence (rapide, bon premier passage).
  3. fit_and_remove_sinusoid() : ajuste un modèle sinusoïdal
     (amplitude + fréquence + phase, éventuellement variable) sur les
     régions SANS pic Raman, puis le soustrait sur tout le spectre
     (plus précis, recommandé pour la correction finale).

Usage typique : lance d'abord detect_fringe_frequency() pour voir le
spectre de puissance et repérer le pic des franges, puis utilise cette
fréquence comme point de départ pour fit_and_remove_sinusoid().
"""

import numpy as np
from extract_zone import traiter_acquisitions, extraire_fichiers_jour_0, extraire_fichiers_j2_fixe, extraire_fichiers_jour_2,  extraire_fichiers_jour_4, extraire_fichiers_jours_8_11, extraire_fichiers_jour8_frais
from scipy.fft import rfft, irfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def detect_fringe_frequency(spectrum, x=None, exclude_low_freq=0.02, plot=True):
    """
    Identifie la fréquence spatiale dominante d'une modulation périodique
    (etaloning) dans un spectre, via analyse FFT.

    Parameters
    ----------
    spectrum : array-like
        Intensités du spectre (en fonction du pixel ou du nombre d'onde).
    x : array-like, optional
        Axe x correspondant (pixel ou cm-1). Utilisé seulement pour les
        graphiques.
    exclude_low_freq : float
        Fraction des basses fréquences à ignorer (0.02 = 2%), pour ne pas
        confondre la tendance générale du spectre (large, basse fréquence)
        avec les franges (plus haute fréquence, périodiques).
    plot : bool
        Affiche le spectre et son spectre de puissance FFT.

    Returns
    -------
    dominant_freq : float
        Fréquence dominante détectée (en cycles / échantillon).
    period_pixels : float
        Période correspondante, en nombre de pixels.
    """
    n = len(spectrum)
    spectrum = np.asarray(spectrum, dtype=float)

    # on retire la tendance moyenne / lente pour isoler les hautes fréquences
    detrended = spectrum - np.polyval(np.polyfit(np.arange(n), spectrum, 3), np.arange(n))

    fft_vals = rfft(detrended)
    freqs = rfftfreq(n)
    power = np.abs(fft_vals)

    # on ignore les toutes basses fréquences (tendance résiduelle)
    low_cut = int(exclude_low_freq * n)
    power_search = power.copy()
    power_search[:low_cut] = 0

    peak_idx = np.argmax(power_search)
    dominant_freq = freqs[peak_idx]
    period_pixels = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

    if plot:
        import matplotlib.pyplot as plt
        x_axis = x if x is not None else np.arange(n)

        fig, axes = plt.subplots(2, 1, figsize=(9, 6))
        axes[0].plot(x_axis, spectrum, lw=0.8)
        axes[0].set_title("Spectre brut")
        axes[0].set_xlabel("pixel / cm-1")

        axes[1].plot(freqs, power)
        axes[1].axvline(dominant_freq, color="red", ls="--",
                         label=f"pic détecté: période ≈ {period_pixels:.1f} px")
        axes[1].set_title("Spectre de puissance (FFT)")
        axes[1].set_xlabel("fréquence spatiale (cycles/pixel)")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig("/home/claude/fft_diagnostic.png", dpi=120)
        plt.close()

    return dominant_freq, period_pixels


def remove_etalon_fft(spectrum, freq_low, freq_high):
    """
    Filtre coupe-bande simple : met à zéro les composantes FFT dans
    l'intervalle [freq_low, freq_high] (cycles/échantillon), là où
    se trouvent les franges d'etaloning.

    À utiliser en première approche / diagnostic rapide. Attention :
    si un pic Raman partage la même fréquence spatiale que les franges,
    il sera aussi atténué. Préférer fit_and_remove_sinusoid() pour un
    résultat plus propre.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n = len(spectrum)

    fft_vals = rfft(spectrum)
    freqs = rfftfreq(n)

    mask = (freqs >= freq_low) & (freqs <= freq_high)
    fft_vals[mask] = 0

    return irfft(fft_vals, n=n)


def _sinusoid_model(x, amplitude, freq, phase, offset):
    return offset + amplitude * np.sin(2 * np.pi * freq * x + phase)


def fit_and_remove_sinusoid(spectrum, initial_freq, raman_peak_mask=None):
    """
    Ajuste un modèle sinusoïdal simple sur le spectre (idéalement en
    excluant les régions où se trouvent de vrais pics Raman) puis le
    soustrait de l'ensemble du spectre.

    Parameters
    ----------
    spectrum : array-like
        Le spectre à corriger.
    initial_freq : float
        Estimation initiale de la fréquence des franges (cycles/pixel),
        typiquement obtenue via detect_fringe_frequency().
    raman_peak_mask : array-like of bool, optional
        Masque de même longueur que spectrum, True là où il y a un vrai
        pic Raman (à exclure du fit). Si None, le fit se fait sur tout
        le spectre (moins précis si les pics sont larges/intenses).

    Returns
    -------
    corrected : ndarray
        Spectre après soustraction du modèle sinusoïdal ajusté.
    fringe_model : ndarray
        Le modèle de frange lui-même (utile pour vérification visuelle).
    params : tuple
        (amplitude, freq, phase, offset) ajustés.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n = len(spectrum)
    x = np.arange(n)

    if raman_peak_mask is not None:
        fit_x = x[~raman_peak_mask]
        fit_y = spectrum[~raman_peak_mask]
    else:
        fit_x, fit_y = x, spectrum

    amp0 = (np.percentile(fit_y, 95) - np.percentile(fit_y, 5)) / 2
    offset0 = np.mean(fit_y)
    p0 = [amp0, initial_freq, 0.0, offset0]

    try:
        params, _ = curve_fit(_sinusoid_model, fit_x, fit_y, p0=p0, maxfev=10000)
    except RuntimeError:
        print("Le fit n'a pas convergé — essaie d'ajuster initial_freq ou le masque.")
        return spectrum, np.zeros_like(spectrum), tuple(p0)

    fringe_model = _sinusoid_model(x, *params)
    corrected = spectrum - fringe_model + params[3]  # on garde l'offset (ligne de base)

    return corrected, fringe_model, params


def auto_detect_raman_peaks(spectrum, prominence=None):
    """
    Petite aide : détecte automatiquement les positions probables de
    vrais pics Raman (pics étroits et proéminents) pour construire un
    masque à passer à fit_and_remove_sinusoid(). À valider visuellement,
    ce n'est qu'une heuristique de départ.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if prominence is None:
        prominence = 0.1 * (spectrum.max() - spectrum.min())

    peaks, properties = find_peaks(spectrum, prominence=prominence, width=(None, None))
    mask = np.zeros(len(spectrum), dtype=bool)

    widths = properties["widths"]
    for peak, width in zip(peaks, widths):
        half_width = int(np.ceil(width))
        lo = max(0, peak - half_width)
        hi = min(len(spectrum), peak + half_width + 1)
        mask[lo:hi] = True

    return mask


if __name__ == "__main__":

    extracteur = {
    'jour0':   extraire_fichiers_jour_0,
    'jour_2':   extraire_fichiers_j2_fixe,
    'jour4':   extraire_fichiers_jour_4,
    'jour_8':  extraire_fichiers_jours_8_11,
    'jour_11': extraire_fichiers_jours_8_11,
}

    liste_fichiers = extracteur['jour0']('jour0', 'petri1', 'souris1','echantillon1', 'zone1')

    w, _, i = traiter_acquisitions(liste_fichiers)

    # 1. détecter la fréquence des franges
    freq, period = detect_fringe_frequency(i, plot=False)
    print(f"Fréquence détectée: {freq:.5f} cycles/pixel, période ≈ {period:.1f} pixels")

    # 2. masquer les vrais pics Raman
    mask = auto_detect_raman_peaks(i)

    # 3. fitter et soustraire le modèle sinusoïdal
    corrected, fringe_model, params = fit_and_remove_sinusoid(
        i, initial_freq=freq, raman_peak_mask=mask
    )
    print(f"Paramètres ajustés (amplitude, freq, phase, offset): {params}")

    # sauvegarde d'un graphique de contrôle
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(w, i, lw=0.8)
    axes[0].set_title("Spectre simulé (avec franges)")
    axes[1].plot(w, fringe_model, color="orange")
    axes[1].set_title("Modèle de frange ajusté")
    axes[2].plot(w, corrected, lw=0.8, color="green")
    axes[2].plot(w, i + 20, lw=0.8, color="black", ls="--", alpha=0.5,
                 label="signal Raman vrai (référence)")
    axes[2].set_title("Spectre corrigé vs signal vrai")
    axes[2].legend()
    plt.tight_layout()
    plt.show()