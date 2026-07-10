import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import os
from orpl.baseline_removal import bubblefill
import glob
from scipy.optimize import lsq_linear
import numpy as np
dossier_verre = r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_2\spectre du verre"
liste_fichiers_verre =  sorted(glob.glob(os.path.join(dossier_verre)))
print(liste_fichiers_verre)
