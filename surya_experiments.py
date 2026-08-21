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
import unittest
from multiprocessing import Lock, Queue
from collections import deque
from threading import Thread
from contextlib import contextmanager
import unicodedata
import shutil

DEBUG = True

def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def extract_properties_from_path(root, file_relative_path):
    """
    We match the text of the path with various patterns to extract the metadata,
    which we return in a dictionary

    """
    properties = {}

    file_path = str(Path(root) / Path(file_relative_path) )

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
    else:
        properties['test'] = False

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

    properties['file'] = str(file_relative_path)

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

def extract_header_from_relative_path(root, file_path):
    return extract_header_from_path(Path(root) / Path(file_path))

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

def get_mask(df, mask_as_dict):
    mask = pd.Series(True, index=df.index)
    for key, value in mask_as_dict.items():
        if key not in df.columns:
            continue
        mask &= (df[key].notna() & (df[key] == value))

    return mask

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
        if len(list_rows) == 0:
            return df

        if len(set(list_rows)) == len(list_rows):
            raise ValueError(f"Les 'indice1' sont uniques: rien a renumeroter {list_rows}")

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

    print_debug(f"\n\n== 4. Meme probleme que #2 et #3 ==")
    mask_doublons = get_mask(df, {'dizaine': 0, 'exp': 1, 'fixation': 'fixe', 'jour': 4, 'modalite': 'raman', 'petri': 3, 'souris': 2})
    df = renumber_sequentially_in_time(df, mask_doublons)

    print_debug(f"\n\n== 5. Meme probleme que #2 et #3 ==")
    mask_doublons = get_mask(df, {'dizaine': 0, 'exp': 1, 'fixation': 'fixe', 'indice1': 8, 'jour': 8, 'modalite': 'raman', 'petri': 4, 'souris': 5, 'zone': 2})
    if len(df[mask_doublons]) >= 2:
        df = renumber_sequentially_in_time(df, mask_doublons)
        
    print_debug(f"\n\n== 6. Un fichier seul a effacer ==")
    df = df[ df['file'] != "exp_1/jour8/frais/Raman/petri2/petri2_souris3_zone1/20260519_jour6_raman_petri2_souris3_45Gy_zone1_RamanShift__0__11-41-01-478.txt"]

    print_debug(f"\n\n== 7. Renomme exp_2 fixe en exp_3 ==")
    mask_exp2_fixe = get_mask(df, {'exp': 2, 'fixation': 'fixe'})
    df.loc[mask_exp2_fixe, 'exp'] = 3 

    df.to_excel(name+".xlsx", index=False)
    df.to_pickle(name+".pkl")

    return df

def delete_test_data(df):
    assert not df.empty
    df = df[(df['test'] == False)]
    df = df[(df['modalite'] == 'raman')]
    df = df[(df['keyword'] != 'dark')]
    df = df[(df['keyword'] != 'white')]
    df = df[(df['keyword'] != 'adn')]
    df = df[(df['keyword'] != 'black')]
    df = df[(df['keyword'] != 'blanche')]
    df = df[(df['keyword'] != 'anneau')]
    df = df[(df['keyword'] != 'plus_tard')]
    assert not df.empty
    return df
