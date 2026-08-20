import os
import re
import datetime
from pathlib import Path
from collections import defaultdict
import subprocess
import pandas as pd
import time
import hashlib, json
import uuid

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

DEBUG = False

def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def cache_name(params, prefix="cache", ext=".pkl"):
    params['node'] = uuid.getnode()
    s = json.dumps(params, sort_keys=True, default=str)
    h = hashlib.sha256(s.encode()).hexdigest()[:16]
    return f"{prefix}_{h}{ext}"

def get_all_data_file_paths(root, invisible_files=False, progress=True, use_cache=True, max_cache_age_hours=24):
    """
    Get the list of files at a given root directory

    Since this is a lengthy operation on the network, we keep the result and
    use it if use_cache is True

    """

    
    if not Path(root).exists():
        raise ValueError(f"The path {root} does not exist")

    cache_filename = "cache_surya_files.txt"
    cache = Path(root) / Path(cache_filename)

    all_files = []
    
    if use_cache and cache.exists():
        with open(cache, "r", encoding="utf-8") as file_path:
            all_files = file_path.read().splitlines()

    if not all_files:
        print("No cache for files list, reading from disk")
        next_progress_print = time.time() + 2
        for dirpath, dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(dirpath, name)
                if "/." in path and not invisible_files:
                    continue

                if progress and time.time() > next_progress_print:
                    print(".", end = "", flush=True)
                    next_progress_print = time.time() + 2

                if not Path(path).exists():
                    print(f"Warning: {path} is not recognized (probably accented characters)")

                file_relative_path = str(Path(path).relative_to(root))
                all_files.append(file_relative_path)

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

    file_path = str(file_path)

    def to_int_values(properties):
        for key, value in properties.items():
            try:
                if value is None:
                    continue
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
    pattern = r"[\W_\d]S(?:ouris?)?(?P<souris>\d+)\.?(?P<subzone>\d)?"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de la "subzone" si dans le mot echantillon
    pattern = r"echantillon(?P<subzone>\d)"
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
    # Le \W ne match pas le "_", il faut donc [\W_] pour attraper "souris2_0Gy_zone1"
    # et le separateur optionnel pour "zone_1" et "zone 2"
    pattern = r"[\W_\d][Zz]o?n?e?_? ?(?P<zone>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        properties.update(to_int_values(match.groupdict()))

    # Extraction de l'heure d'acquisition et de l'indice1
    pattern = r"__(?P<indice1>\d+)__(?P<heure>\d+)-(?P<minutes>\d+)-(?P<s>\d+)-(?P<ms>\d+)"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        groups = match.groupdict()

        my_time = datetime.time( hour=int(groups['heure']),
                        minute=int(groups['minutes']), 
                        second=int(groups['s']),
                        microsecond=int(groups['ms'])*1000)

        properties['time'] = my_time
        properties['indice1'] = int(groups['indice1'])

    # Extraction de l'indice1 en l'absence de heure d'acquisition
    pattern = r"__(?P<indice1>\d+)__(?P<indice2>\d{5})"
    match = re.search(pattern, file_path,  re.IGNORECASE)
    if match is not None:
        groups = match.groupdict()

        properties['indice1'] = int(groups['indice1'])
        properties['indice2'] = int(groups['indice2'])
    
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
    pattern = r"(?P<keyword>white|blanche|dark|black|verre|gel+ose|anneau|adn|petri_|methanol|pink|\d+\s*min\s*plus\s*tards?)"
    match = re.search(pattern, file_path, re.IGNORECASE)
    if match is not None:
        groups = match.groupdict()
        if "gellose" in groups.values():
            groups['keyword'] = "gelose"
        if 'petri_' in groups.values():
            groups['keyword'] = 'petri'
        if groups['keyword'] is not None and re.match(r"\d+\s*min\s*plus\s*tards?", groups['keyword'], re.IGNORECASE):
            groups['keyword'] = 'plus_tard'

        properties.update(to_int_values(groups))

    properties['file'] = str(Path(file_path).relative_to(root))

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

def get_files_metadata(root, all_files, header = True, extended = True, use_cache = True):
    all_properties = []

    cache_filename = "cache_surya_files_metadata.pkl"
    cache_path = Path(root) / Path(cache_filename)

    df = pd.DataFrame()
    df_file_set = {}
    if use_cache and cache_path.exists():
        df = pd.read_pickle(cache_path)
        df = df.reset_index(drop=True) 
        df_file_set = { str(file) for file in df['file']}

    # We show progress every 2 seconds
    next_progress_print = time.time() + 2

    for i, file_relative_path in enumerate(all_files):
        if file_relative_path in df_file_set:
            continue

        if time.time() > next_progress_print:
            print(f"Processing file {i} of {len(all_files)}")
            next_progress_print += 2

        file_path = Path(root) / file_relative_path

        properties = extract_properties_from_path(file_path)

        # This is slow: they must open and read the file_path
        if header:
            header_properties = extract_header_from_path(file_path)
            properties.update(header_properties)

        if extended:
            extended_properties = extract_extended_properties_from_path(file_path)
            properties.update(extended_properties)
        
        single_row = pd.DataFrame([properties])
        df = pd.concat([df, single_row], ignore_index=True)

    # We can force the type of certain columns to be clean
    convert_to_int = {"exp","petri","jour", "souris", "indice1", "indice2", "dose", "test", "zone", "subzone","batch"}
    df = df.astype({c: "Int64" for c in convert_to_int if c in df.columns})


    summary = "surya-dataset-description"    
    df.to_excel(summary+".xlsx", index=False)
    print_debug(f"Summary written to {summary}")

    df.to_pickle(cache_path)
    print_debug(f"Cache updated and written to {cache_path}")

    return df, summary

def get_mask(df, mask_as_dict):
    mask = pd.Series(True, index=df.index)
    for key, value in mask_as_dict.items():
        if key not in df.columns:
            continue
        mask &= (df[key].notna() & (df[key] == value))

    return mask

def validate_unique_metadata(df, ignore=("time", "indice1", "file", "size_in_bytes"), verbose=True):
    """
    Verifie que les metadata de chaque fichier sont uniques.

    Pour chaque ligne, on rassemble les metadata dans un dictionnaire, on
    enleve les colonnes qui sont toujours differentes (time, indice1, file) puis
    on verifie que la signature qui reste n'apparait qu'une seule fois.

    Retourne un dictionnaire {signature: [liste des fichiers]} pour les
    signatures qui apparaissent plus d'une fois (donc les doublons).
    """
    colonnes = [c for c in df.columns if c not in ignore and not c.startswith("Spectrum:")]

    signatures = {}
    for i, row in df[colonnes].iterrows():
        # Les metadata de cette ligne, sans les valeurs manquantes
        metadata = {k: v for k, v in row.items() if pd.notna(v)}
        # Un dict n'est pas hashable: on en fait un tuple trie pour la clef
        signature = tuple(sorted(metadata.items(), key=lambda kv: str(kv[0])))
        signatures.setdefault(signature, []).append(i)

    doublons = {sig: idx for sig, idx in signatures.items() if len(idx) > 1}

    if verbose:
        n_doublons = sum(len(idx) for idx in doublons.values())
        if not doublons:
            print_debug(f"Toutes les metadata sont uniques ({len(df)} fichiers, colonnes: {colonnes})")
        else:
            print(f"{len(doublons)} signatures non-uniques touchant {n_doublons} fichiers:")
            for signature, indices in doublons.items():
                if len(indices) % 5 != 0:
                    print(f"\n #{len(indices)} {dict(signature)}")
                    for i in indices:
                        print(f"    {df.loc[i, 'file']}")

    # On retourne les fichiers plutot que les indice1, plus utile pour le diagnostic
    return {sig: df.loc[idx, "file"].tolist() for sig, idx in doublons.items()}

def helper_find_root_directory():
    options = [ "/Volumes/Labdata/dcclab/surya",
                r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya",
                "/home/dccadmin/labdata/dcclab/surya",
                "."]
    for path in options:
        if Path(path).exists() and ("surya" in str(Path(path).resolve()).lower()):
            return path


def add_additional_experimental_info(dataframe, name="surya-dataset-description" ):
    from config import CONFIG1 as config1, CONFIG2 as config2

    # adding data in panda dataframe
    for batch, petris in config1.items():
        for petri, (echantillon, dose, type_) in petris.items():
            num_batch = int(re.search(r'\d+', batch).group())
            num_petri = int(re.search(r'\d+', petri).group())
            masque = (dataframe['exp'] == 2) & (dataframe['batch'] == num_batch) & (dataframe['petri'] == num_petri)
            dataframe.loc[masque, 'dose'] = dose
            dataframe.loc[masque, 'sexe'] = type_[0].lower()
            dataframe.loc[masque, 'traitement'] = 'NT' not in type_

    for jour, petris in config2.items():
        for petri, (doses, souris_data) in petris.items():
            num_petri = int(re.search(r'\d+', petri).group())
            num_jour = int(re.search(r'\d+', jour).group())
            dose = int(re.search(r'\d+', doses).group())
            traitement = '+' in doses

            masque1 = (dataframe['exp'] == 1) & (dataframe['jour'] == num_jour) & (dataframe['petri'] == num_petri)
            dataframe.loc[masque1, 'dose'] = dose
            dataframe.loc[masque1, 'traitement'] = traitement

            for souris, info in souris_data.items():
                num_souris = int(re.search(r'\d+', souris).group())
                masque2 = masque1 & (dataframe['souris'] == num_souris)
                sexe = 'f' if num_souris in (1, 2, 3) else 'm'
                dataframe.loc[masque2, 'sexe'] = sexe

    dataframe.to_excel(name+".xlsx", index=False)
    dataframe.to_pickle(name+".pkl")

    return dataframe
    

def fix_acquisition_errors(df, name="surya-dataset-description"):
    """
    Some errors occured during acquisition and were noted in the experimenter's labbook.
    They are corrected here (not in the raw data)
    """

    def renumber_sequentially_in_time(df, mask):
        list_rows = df[mask].sort_values('indice1')['indice1']
        if len(set(list_rows)) == len(list_rows):
            raise ValueError("Les 'indice1' sont uniques: rien a renumeroter")

        indices = df[mask].sort_values('time').index
        for i, idx in enumerate(indices):
            df.loc[idx, 'indice1'] = i

        list_rows = df[mask].sort_values('indice1')['indice1']
        if len(set(list_rows)) != len(list_rows):
            raise ValueError("Warning: La renumerotation n'a pas fonctionne")
        else:
            print_debug("'indice1' rewritten sequentially")

        return df

    # Pour aider, on calcule les dizaines des indice1 (permettra de reconnaitre les problemes)
    masque = df['indice1'].notna()
    df.loc[masque, 'dizaine'] = df['indice1'] // 10


    print_debug(f"\n\n== 1. Gestion des erreurs d'acquisition dans exp 2, batch 1, souris 48 (fichiers copies par erreur dans petri 5 et 7) ==")
    count_before = len(df)
    mask_a_enlever = get_mask(df, {"exp":2, "souris":48, "batch":1, "petri":7})
    df = df[~mask_a_enlever]
    mask_a_enlever = get_mask(df, {"exp":2, "souris":48, "batch":1, "petri":5})
    df = df[~mask_a_enlever]
    count_after = len(df)
    print_debug(f"  Avant/apres : {count_before}/{count_after}, {count_before-count_after} effaces")

    print_debug(f"\n\n== 2. Exp1, jour 2, petri 1, souris 1: l'indice d'acquisition commence a 1, et recommence ensuite a 0. On renomme sequentillement ==")
    mask_doublons = get_mask(df, { 'exp': 1, 'fixation': 'fixe', 'jour': 2, 'keyword': 'verre', 'modalite': 'raman', 'petri': 1, 'souris': 1})
    df = renumber_sequentially_in_time(df, mask_doublons)

    print_debug(f"\n\n== 3. Meme probleme que #2 mais exp 1, jour 4 petri 3 souris 1 ==")
    mask_doublons = get_mask(df, {'dizaine': 0, 'exp': 1, 'fixation': 'fixe', 'jour': 4, 'modalite': 'raman', 'petri': 3, 'souris': 1})
    df = renumber_sequentially_in_time(df, mask_doublons)

    print_debug(f"\n\n== 4. Meme probleme que #2 et #3 mais exp 1, jour 4 petri 3 souris 2==")
    mask_doublons = get_mask(df, {'dizaine': 0, 'exp': 1, 'fixation': 'fixe', 'jour': 4, 'modalite': 'raman', 'petri': 3, 'souris': 2})
    df = renumber_sequentially_in_time(df, mask_doublons)

    try:
        print_debug(f"\n\n== 5. Meme probleme que #2 et #3 mais exp 1, jour 4 petri 3 souris 2==")
        mask_doublons = get_mask(df, {'dizaine': 0, 'exp': 1, 'fixation': 'fixe', 'indice1': 8, 'jour': 8, 'modalite': 'raman', 'petri': 4, 'souris': 5, 'zone': 2})
        df = renumber_sequentially_in_time(df, mask_doublons)
    except:
        pass

        
    print_debug(f"\n\n== 6. Un fichier seul a effacer ==")
    df = df[ df['file'] != "exp_1/jour8/frais/Raman/petri2/petri2_souris3_zone1/20260519_jour6_raman_petri2_souris3_45Gy_zone1_RamanShift__0__11-41-01-478.txt"]

    print_debug(f"\n\n== 7. Renomme exp_2 fixe en exp_3 ==")
    mask_exp2_fixe = get_mask(df, {'exp': 2, 'fixation': 'fixe'})
    df.loc[mask_exp2_fixe, 'exp'] = 3 

    df.to_excel(name+".xlsx", index=False)
    df.to_pickle(name+".pkl")

    return df


def delete_test_data(df):
    df = df[(df['test'].isna())]
    df = df[(df['modalite'] == 'raman')]
    df = df[(df['keyword'] != 'dark')]
    df = df[(df['keyword'] != 'white')]
    df = df[(df['keyword'] != 'adn')]
    df = df[(df['keyword'] != 'black')]
    df = df[(df['keyword'] != 'blanche')]
    df = df[(df['keyword'] != 'anneau')]
    df = df[(df['keyword'] != 'plus_tard')]
    return df

def get_surya_dataframe():
    root =  helper_find_root_directory()
    print(f"Root path is: {root}")
  
    all_files = get_all_data_file_paths(root, use_cache= True)
    all_files = [ file_path for file_path in all_files if file_path.endswith('txt') ]  # Keep only data files
    all_files = [ file_path for file_path in all_files if "old" not in file_path]      # exp_2_old is not useful, we remove it
    all_files = [ file_path for file_path in all_files if "archives" not in file_path] # archives is not useful, we remove it 

    # Get metadata about the data (from path name, from spectrum header and from actual file info)
    df, summary = get_files_metadata(root, all_files, header = True, extended=True, use_cache = True)

    df = delete_test_data(df)

    df = add_additional_experimental_info(df, summary) #add information manually    

    df = fix_acquisition_errors(df)

    print_debug("\n\n== Verification de l'unicite des metadata ==\n")
    doublons = validate_unique_metadata(df, ignore=("time", "file","size_in_bytes"))
    if doublons:
        print(doublons)

    return df

def build_mask(df, criteria):
    uid_fields = {"exp", "souris", "cote", "zone"}

    mask = None
    for key, value in criteria.items():
        if mask is None:
            mask = (df[key] == value)
        else:
            mask &= (df[key] == value)        

    remaining_fields = uid_fields.difference(set(criteria.keys()))
    
    return mask, df[mask][list(remaining_fields)]

if __name__ == "__main__":

    my_cache = 'my_cache.pkl'
    if Path(my_cache).exists():
        df = pd.read_pickle(my_cache)
    else:
        df = get_surya_dataframe()
        df.to_pickle(my_cache)

    # uid1_fields = {"exp", "souris", "jour", "petri", "zone"}

    mask, remaining_values = build_mask(df, {'exp':2, 'souris':27})

    print(remaining_values)