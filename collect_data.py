import os
import re
import datetime
from pathlib import Path
import subprocess
import pandas as pd
import time

"""
This code will read all the spectral files from a root directory and
extract all the metadata about the spectra (mouse, petri, dose, etc...)

At the bottom of the file, you will find example code that gets run when running this file.

if __name__ == "__main__":
    # Start here

"""
def get_all_files(root, invisible_files = False, progress = True, use_cache = True):
    """
    Get the list of files at a given root directory

    """
    if not Path(root).exists():
        raise ValueError(f"The path {root} does not exist")

    cache = Path("all_files-cache.txt")
    all_files = []
    
    if use_cache and cache.exists():
        with open(cache, "r", encoding="utf-8") as file:
            all_files = file.read().splitlines()
            if all_files and all_files[0].startswith(root):
                print(f"Cache read from {cache}")
            else:
                all_files = []

    if not all_files:
        print("No cache, reading from disk")
        next_progress_print = time.time() + 2
        for dirpath, dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(dirpath, name)
                if "/." in path and not invisible_files:
                    continue

                if progress and time.time() > next_progress_print:
                    print(".", end = "")
                    next_progress_print = time.time() + 2

                if not Path(path).exists():
                    print(f"Warning: {path} is not recognized (probably accented characters)")

                all_files.append(path)
        print(".")

        if use_cache:
            with open(cache, "w", encoding="utf-8") as file:
                for line in all_files:
                    file.write(line + "\n")
                print(f"Cache saved to {cache}. Delete the file if you organize the data differently")

    return all_files

def extract_properties_from_path(file):
    """
    We match the text of the path with various patterns to extract the metadata,
    which we return in a dictionary

    """
    properties = {}

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

    properties['file'] = file

    return properties

def extract_extended_properties_from_path(file):
    """
    Get more information about the file, but slow.
    Returns a dictionary with the properties
    """
    extended_properties = {}

    # Fetch file stats (slow)
    file_info = Path(file).stat()
    
    extended_properties['size_in_bytes'] = file_info.st_size
    # Others possible

    return extended_properties

def extract_header_from_spectral_file(file):
    """
    Since we use Ocean Optics Raman QEPro, we read the header of the 
    file a extract the metadata from the header, which we return in a
    dictionary

    """
    properties = {}
    with open(file,"r", encoding="utf-8", errors="ignore") as file:
        first_line = file.readline()
        if first_line.startswith("Data from"):
            # It is a Raman spectral file
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    entry = line.split(":", 1)
                    if len(entry) == 2 :
                        properties[f"Spectrum:{entry[0]}"] = entry[1]
                if ">>>>>Begin Spectral Data<<<<<" in line:
                    break

    return properties


if __name__ == "__main__":
    root = "/Volumes/labdata/dcclab/surya" # Pour DCC
    # root = r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya" # Pour Chloe

    all_files = get_all_files(root)
    all_files = [ file for file in all_files if file.endswith('txt') ]  # Keep only data files
    all_files = [ file for file in all_files if "old" not in file]      # exp_2_old is not useful, we remove it
    all_files = [ file for file in all_files if "archives" not in file] # archives is not useful, we remove it 

    all_properties = []

    # We show progress every 2 seconds
    next_progress_print = time.time() + 2

    for i, file in enumerate(all_files):
        if time.time() > next_progress_print:
            print(f"{i} of {len(all_files)}")
            next_progress_print += 2

        properties = extract_properties_from_path(file)

        # This is slow: they must open and read the file
        # header_properties = extract_header_from_spectral_file(file)
        # properties.update(header_properties)

        # extended_properties = extract_extended_properties_from_path(file)
        # properties.update(extended_properties)

        all_properties.append(properties)

    # Put into a Panda dataframe and save everything for review
    df = pd.DataFrame(all_properties)
    summary = "surya-dataset-description"
    df.to_excel(summary+".xlsx", index=False)
    df.to_pickle(summary+".pkl")

    print(f"Summary written to {summary}")
