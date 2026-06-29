import matplotlib.pyplot as plt
from extract_data import traiter_acquisitions
from pathlib import Path


racine = Path(r"C:\Users\chloe\OneDrive - Université Laval\Stage_été_2026\Projet_Surya\acquisition_données_Surya\jour_11\Raman\spectre_verre")
ws, i = traiter_acquisitions(list(racine.glob("*")))
w_lambda = [1/(1/785 - w) for w in ws]


plt.figure(figsize=(10, 6))
plt.plot(w_lambda, i, label='Spectre Raman')
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel('Intensité')
plt.title("Spectre Raman d'une lamelle de microscope")
plt.show()