import os
import re
import datetime
from pathlib import Path
import subprocess
import pandas as pd
import time

"""
This code will read all the spectral files from a root directory and
extract all the metadata about the spectra (mouse, petri, dose, etc...).

It will store everything in a Panda Dataframe, which is similar to an Excel file.
It will try to avoid recomputing everything every time and will keep a "cached" copy
and use it whenever it is possible.

At the bottom of the file, you will find example code that gets run when running this file.
You will find many examples on how to filter a DataFrame.

if __name__ == "__main__":
    # Start here

"""
def get_all_data_file_paths(root, invisible_files = False, progress = True, use_cache = True):
    """
    Get the list of files at a given root directory

    Since this is a lengthy operation on the network, we keep the result and
    use it if use_cache is True

    """
    if not Path(root).exists():
        raise ValueError(f"The path {root} does not exist")

    cache = Path("all_files-cache.txt")
    all_files = []
    
    if use_cache and cache.exists():
        with open(cache, "r", encoding="utf-8") as file_path:
            all_files = file_path.read().splitlines()
            if all_files and all_files[0].startswith(root):
                print(f"Cache read from {cache}")
            else:
                all_files = []

    if not all_files:
        print("No cache for files list, reading from disk")
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
            with open(cache, "w", encoding="utf-8") as file_path:
                for line in all_files:
                    file_path.write(line + "\n")
                print(f"Cache saved to {cache}. Delete the file if you organize the data differently")

    return all_files

def extract_properties_from_path(file_path):
    """
    We match the text of the path with various patterns to extract the metadata,
    which we return in a dictionary

    """
    properties = {}

    def to_int_values(properties):
        for key, value in properties.items():
            try:
                if str(int(value)) == value:
                    properties[key] = int(value)
            except:
                properties[key] = value.lower()

        return properties


    # Extraction de exp
    pattern = r"exp_?(?P<exp>\d)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de petri
    pattern = r"petri(?P<petri>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction du jour
    pattern = r"jour(?P<jour>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de la souris
    pattern = r"[\W_\d]S(?:ouris?)?(?P<souris>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de la modalite
    pattern = r"(?P<modalite>raman|drs|speckles)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de la dose
    pattern = r"(?P<dose>\d+)Gy"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de la batch
    pattern = r"batch#(?P<batch>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de la zone
    pattern = r"\W[Zz]o?n?e?(?P<zone>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de l'heure d'acquisition'
    pattern = r"__(?P<index>\d+)__(?P<heure>\d+)-(?P<minutes>\d+)-(?P<s>\d+)-(?P<ms>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        groups = match.groupdict()

        my_time = datetime.time( hour=int(groups['heure']),
                        minute=int(groups['minutes']), 
                        second=int(groups['s']),
                        microsecond=int(groups['ms'])*1000)

        properties['time'] = my_time
        properties['index'] = int(groups['index'])

    # Extraction de la hauteur, si presente
    pattern = r"\WHauteur(?P<hauteur>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))


    # Extraction de la fixation
    pattern = r"(?P<fixation>frais|fixe)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction du cote
    pattern = r"-(?P<cote>[DG])-"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction du mot test, si present
    pattern = r"tests?"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties['test'] = True

    # Extraction de certains keywords, si present
    pattern = r"(?P<keyword>white|blanche|dark|black|verre|gel+ose|anneau|adn)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        groups = match.groupdict()
        if "gellose" in groups.values():
            groups['keyword'] = "gelose"

        properties.update(to_int_values(groups))

    properties['file'] = file_path

    return properties

def extract_extended_properties_from_path(file_path):
    """
    Get more information about the file_path, but slow.
    Returns a dictionary with the properties
    """
    extended_properties = {}

    # Fetch file_path stats (slow)
    try:
        file_info = Path(file_path).stat()
        
        extended_properties['size_in_bytes'] = file_info.st_size
        # Others possible
    except Exception as e:
        pass # We just give up if unable to do it

    return extended_properties

def extract_header_from_path(file_path):
    """
    Since we use Ocean Optics Raman QEPro, we read the header of the 
    file_path a extract the metadata from the header, which we return in a
    dictionary

    """
    properties = {}
    try:
        with open(file_path,"r", encoding="utf-8", errors="ignore") as file:
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

    except Exception as e:
        print(f"Warning: {file_path} is not recognized (probably accented characters)")

    return properties

def get_files_metadata(all_files, header = True, extended = True, use_cache = True):
    all_properties = []

    # If we can use the cache, we do
    summary = "surya-dataset-description"
    if use_cache and Path(summary+".pkl").exists():
        df = pd.read_pickle(Path(summary+".pkl"))
        if set(all_files).issubset(df["file"]):
            return df
        # If not, we need to retrieve it

    # We show progress every 2 seconds
    next_progress_print = time.time() + 2

    for i, file_path in enumerate(all_files):
        if time.time() > next_progress_print:
            print(f"Processing file {i} of {len(all_files)}")
            next_progress_print += 2

        properties = extract_properties_from_path(file_path)

        # This is slow: they must open and read the file_path
        if header:
            header_properties = extract_header_from_path(file_path)
            properties.update(header_properties)

        if extended:
            extended_properties = extract_extended_properties_from_path(file_path)
            properties.update(extended_properties)
        all_properties.append(properties)

    # A panda dataframe is like an excel file with column titles
    # Put into a Panda dataframe and save everything for review

    df = pd.DataFrame(all_properties)

    # We can force the type of certain columns to be clean
    convert_to_int = {"exp","petri","jour", "souris", "index", "dose", "test", "zone", "batch"}
    df = df.astype({c: "Int64" for c in convert_to_int if c in df.columns})

    df.to_excel(summary+".xlsx", index=False)
    df.to_pickle(summary+".pkl")

    print(f"Summary written to {summary}")
    return df

def helper_find_root_directory():
    options = [ "/Volumes/Labdata/dcclab/surya",
                r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya",
                "/home/dccadmin/labdata/dcclab/surya",
                "."]
    for path in options:
        if Path(path).exists() and ("surya" in str(Path(path).resolve()).lower()):
            return path

if __name__ == "__main__":
    root =  helper_find_root_directory()

    all_files = get_all_data_file_paths(root, use_cache=True)
    all_files = [ file_path for file_path in all_files if file_path.endswith('txt') ]  # Keep only data files
    all_files = [ file_path for file_path in all_files if "old" not in file_path]      # exp_2_old is not useful, we remove it
    all_files = [ file_path for file_path in all_files if "archives" not in file_path] # archives is not useful, we remove it 

    # A panda dataframe is like an excel file with column titles, it is the best structure for data
    # Put into a Panda dataframe and save everything for review
    df = get_files_metadata(all_files, header = False, extended=False, use_cache = False)


    # Manual additions to metadata from labbook
    masque = (df['exp'] == 2) & (df['batch'] == 1) & (df['petri'] == 1)
    df.loc[masque, 'dose'] = 45
    df.loc[masque, 'sexe'] = 'f'
    df.loc[masque, 'traitement'] = False

    # How to manipulate a panda Dataframe:
    
    print("\n\n== Example : list all columns ==\n")
    print(df.columns)

    print("\n\n== Example : list exp only ==\n")
    print(df.exp)
    print(df['exp'])

    print("\n\n== Example :  exp_1 only ==\n")
    print(df[df['exp'] == 1])

    print("\n\n== Example :  exp_1 only, jour 8 ==\n")
    print(df[ (df['exp'] == 1)  & (df['jour'] == 8) ])    

    print("\n\n== Example :  exp_1 only, Raman ==\n")
    print(df[ (df['exp'] == 1)  & (df['modalite'] == 'raman') ])        

    print("\n\n== Example :  exp_1 only, Raman with dose ==\n")
    print(df[ (df['exp'] == 1)  & (df['modalite'] == 'raman') & (df['dose'].notna()) ])

    print("\n\n== Example :  exp_1 only, Raman with dose ==\n")
    print(df[ (df['exp'] == 1)  & (df['modalite'] == 'raman') & (df['dose'].notna()) ])

    print("\n\n== Example :  Frais seulement ==\n")
    print(df[ (df['fixation'] == 'frais') ])

    print("\n\n== Example : Extraire batch 1, petri 1  ==\n")
    print(df[ (df['exp'] == 2) & (df['batch'] == 1 ) & (df['petri'] == 1 ) ]['souris'])

    print("\n\n== List of all values per key ==\n")
    for col in df.columns:
        if col in ['time','file','size_in_bytes'] or col.startswith("Spectrum:"):
            continue
        print(f"{col:20s} ({df[col].nunique()} valeurs) : {sorted(df[col].dropna().unique().tolist(), key=str)}")
