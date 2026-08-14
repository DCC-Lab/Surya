import os
import re
import datetime
from pathlib import Path
import subprocess
import pandas as pd
import time

def get_all_files(root):
    cache = Path("all_files-cache.txt")
    all_files = []
    
    if cache.exists():
        with open(cache, "r", encoding="utf-8") as file:
            all_files = file.read().splitlines()
    else:
        for root, dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(root, name)
                all_files.append(path)

        with open(cache, "w", encoding="utf-8") as file:
            for line in all_files:
                file.write(line + "\n")

    return all_files

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

        my_time = datetime.time( hour=int(groups['heure']),
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

    # # Fetch file stats
    # file_info = Path(file).stat()
    
    # properties['size_in_bytes'] = file_info.st_size
    # properties['modification_time'] = datetime.datetime.fromtimestamp(file_info.st_mtime)
    properties['file'] = file

    return properties

def extract_extended_properties_from_path(file):
    extended_properties = {}

    # Fetch file stats (slow)
    file_info = Path(file).stat()
    
    extended_properties['size_in_bytes'] = file_info.st_size
    extended_properties['modification_time'] = datetime.datetime.fromtimestamp(file_info.st_mtime)

    return extended_properties

if __name__ == "__main__":
    root = "/Volumes/Labdata/dcclab/surya"
    all_files = get_all_files(root)
    all_files = [ file for file in all_files if file.endswith('txt') ]
    all_files = [ file for file in all_files if "/." not in file]
    all_files = [ file for file in all_files if "old" not in file]
    all_files = [ file for file in all_files if "archives" not in file]

    all_properties = []
    count = len(all_files)
    next_time = time.time() + 2
    for i, file in enumerate(all_files):
        if time.time() > next_time:
            print(f"{i} of {count}")
            next_time += 1

        properties = extract_properties_from_path(file)

        # This is slow: do only if needed (file size and modification times)
        #
        # extended_properties = extract_extended_properties_from_path(file)
        # properties.update(extended_properties)

        all_properties.append(properties)

    df = pd.DataFrame(all_properties)
    df.to_excel("surya-dataset.xlsx", index=False)

