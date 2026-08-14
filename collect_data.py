import os
import re
from datetime import datetime, time
from pathlib import Path
import subprocess
import pandas as pd

def get_all_files():
    for root, dirs, files in os.walk("."):
          for name in files:
              yield(os.path.join(root, name))

def extract_properties_from_path(file):
    properties = {}

    # On elimine ce qui n'est pas un fichier de donnees
    if not file.endswith('.txt'):
        return {}


    # Extraction de exp
    pattern = r"exp_?(?P<exp>\d)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction de petri
    pattern = r"petri(?P<petri>\d)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction du jour
    pattern = r"jour(?P<jour>\d+)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction de la souris
    pattern = r"[\W_\d]S(?:ouri).?(?P<souris>\d)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction de la modalite
    pattern = r"(?P<modalite>raman|drs|speckles)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction de la dose
    pattern = r"(?P<dose>\d+)Gy"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())


    # Extraction de la zone
    pattern = r"\W[Zz]o?n?e?(?P<zone>\d+)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction de l'heure d'acquisition'
    pattern = r"__(?P<index>\d+)__(?P<heure>\d+)-(?P<minutes>\d+)-(?P<s>\d+)-(?P<ms>\d+)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        groups = match.groupdict()

        my_time = time( hour=int(groups['heure']),
                        minute=int(groups['minutes']), 
                        second=int(groups['s']),
                        microsecond=int(groups['ms'])*1000)

        properties['time'] = my_time
        properties['index'] = groups['index']

    # Extraction de la hauteur, si presente
    pattern = r"\WHauteur(?P<hauteur>\d+)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())


    # Extraction du cote
    pattern = r"-(?P<cote>[DG])-"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Extraction du mot test, si present
    pattern = r"tests?"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties['test'] = True

    # Extraction du target/type, si present
    
    pattern = r"(?P<target>white|blanche|dark|black|verre|gel+ose|anneau|adn)"
    match = re.search(pattern, file,  re.IGNORECASE)
    if match is not None:
        properties.update(match.groupdict())

    # Fetch file stats
    file_info = Path(file).stat()
    
    properties['size_in_bytes'] = file_info.st_size
    properties['modification_time'] = datetime.fromtimestamp(file_info.st_mtime)
    properties['file'] = file

    return properties

if __name__ == "__main__":
    all_files = get_all_files()
    all_files = [ file for file in all_files if file.endswith('txt') ]
    all_files = [ file for file in all_files if "/." not in file]
    all_files = [ file for file in all_files if "old" not in file]


    # all_properties = { file:extract_properties_from_path(file) for file in all_files }
    all_properties = [ extract_properties_from_path(file) for file in all_files ]

    df = pd.DataFrame(all_properties)
    df.to_excel("summary.xlsx", index=False)

    # for file, properties in all_properties.items():
    #     if "test" in properties:
    #         continue

    #     if "souris" not in properties:
    #         print(file)


