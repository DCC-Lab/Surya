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
import numpy as np

DEBUG = False

def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Every piece of metadata that can be read from the name of a file, written as
# one regular expression per piece. The name between <> is the name of the
# column it will end up in, so adding a new piece of metadata means adding one
# line here and nothing else.
#
# The order matters: a later pattern overwrites what an earlier one found for
# the same name, which is how 'echantillon2' takes precedence over the optional
# subzone of 'souris1.2'.
#
# Values that look like whole numbers become whole numbers, everything else is
# lowercased. A group named 'is_something' is a word that either appears or
# does not, and becomes true or false rather than text. The only thing a
# regular expression cannot express on its own is the acquisition time, whose
# four parts are assembled into one clock time afterwards.

METADATA_PATH_PATTERNS = [
    r"exp_?(?P<exp>\d)",
    r"petri(?P<petri>\d+)",
    r"jour(?P<jour>\d+)",
    # \W does not match '_', hence [\W_], so that 'souris2_0Gy_zone1' is caught
    r"[\W_\d]S(?:ouris?)?(?P<souris>\d+)\.?(?P<subzone>\d)?",
    r"echantillon(?P<subzone>\d)",
    r"(?P<modalite>raman|drs|speckles)",
    r"(?P<dose>\d+)Gy",
    r"batch#(?P<batch>\d+)",
    # the optional separator catches 'zone_1' and 'zone 2' as well as 'zone1'
    r"[\W_\d][Zz]o?n?e?_? ?(?P<zone>\d+)",
    r"__(?P<indice1>\d+)__(?P<heure>\d+)-(?P<minutes>\d+)-(?P<s>\d+)-(?P<ms>\d+)",
    r"__(?P<indice1>\d+)__(?P<indice2>\d{5})",
    r"\WHauteur(?P<hauteur>\d+)",
    r"(?P<fixation>frais|fixe)",
    r"-(?P<cote>[DG])-",
    # A group named 'is_something' records whether the word is there at all,
    # as true or false. It replaces a single 'keyword' column that could only
    # ever hold one word: a file whose name says both 'verre' and 'dark' used
    # to keep whichever came first, and the other was lost. One column each
    # also means a filter reads plainly -- df[~df['is_dark']] -- instead of
    # comparing to a string and hoping the spelling matches.
    r"(?P<is_test>tests?)",
    r"(?P<is_white>white)",
    r"(?P<is_blanche>blanche)",
    r"(?P<is_dark>dark)",
    r"(?P<is_black>black)",
    r"(?P<is_verre>verre)",
    r"(?P<is_gelose>gel+ose)",          # written with one l or two over the years
    r"(?P<is_anneau>anneau)",
    r"(?P<is_adn>adn)",
    r"(?P<is_petri_seul>petri_)",       # 'petri_' alone, not 'petri3'
    r"(?P<is_methanol>methanol)",
    r"(?P<is_pink>pink)",
    r"(?P<is_plus_tard>\d+\s*min\s*plus\s*tards?)",
]


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
            # exp 2 and exp 3 are the same experiment: fix_acquisition_errors()
            # renames the 'fixe' half of exp 2 into exp 3. Accepting both means
            # the doses are assigned whether that renaming has already happened
            # or not, so the order of the finalize() methods no longer matters.
            masque = (dataframe['exp'].isin([2, 3])) & (dataframe['batch'] == num_batch) & (dataframe['petri'] == num_petri)
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

    # index=True: the index holds the file names, they must appear in the export
    dataframe.to_excel(name+".xlsx", index=True)
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
    mask_doublons = get_mask(df, { 'exp': 1, 'fixation': 'fixe', 'jour': 2, 'is_verre': True, 'modalite': 'raman', 'petri': 1, 'souris': 1})
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
    df = df.drop(index="exp_1/jour8/frais/Raman/petri2/petri2_souris3_zone1/20260519_jour6_raman_petri2_souris3_45Gy_zone1_RamanShift__0__11-41-01-478.txt", errors="ignore")

    print_debug(f"\n\n== 7. Renomme exp_2 fixe en exp_3 ==")
    mask_exp2_fixe = get_mask(df, {'exp': 2, 'fixation': 'fixe'})
    df.loc[mask_exp2_fixe, 'exp'] = 3 

    # index=True: the index holds the file names, they must appear in the export
    df.to_excel(name+".xlsx", index=True)
    df.to_pickle(name+".pkl")

    return df

def delete_test_data(df):
    assert not df.empty
    df = df[~df.index.str.startswith(('exp_2_old', 'archives'))]
    df = df[(df['modalite'] == 'raman')]

    # Everything that is not a measurement on a mouse: trial runs, reference
    # spectra of the empty holder, and the ones taken some minutes later. Each
    # is its own true-or-false column, so a file marked by two of these words
    # is caught by either one.
    for word in ('is_test', 'is_dark', 'is_white', 'is_adn',
                 'is_black', 'is_blanche', 'is_anneau', 'is_plus_tard'):
        df = df[~df[word]]
    assert not df.empty
    return df


def helper_find_root_directory():
    """
    Finds the surya data, wherever this particular machine keeps it.

    The same measurements are reached differently depending on who is looking
    and how they mounted the share. Returns None when none of them answers,
    which usually means the network drive is simply not mounted right now.
    """
    options = ["/Volumes/labdata/dcclab/surya",
               r"\\cafeine3.crulrg.ulaval.ca\Goliath\Goliath\labdata\dcclab\surya",
               "/Volume1/Goliath/labdata/dcclab/surya"]

    for path in options:
        if Path(path).exists():
            return path

    return None


class TestSuryaDataset(unittest.TestCase):
    """
    Tests that need the real surya measurements.

    Everything here is about this study in particular: the way its folders are
    named, the doses that were given, the corrections its acquisitions need.
    The tests of DataFiles and of RamanData live with those classes and use
    made-up files, because neither class knows anything about mice.

    These are skipped, not failed, when the network drive is not mounted. A
    skipped test says plainly that nothing was checked; falling back on some
    other folder would instead check the wrong thing and say nothing.
    """

    def setUp(self):
        self.root = helper_find_root_directory()
        if self.root is None:
            self.skipTest("the surya data is not reachable from this machine")

        from datafiles import DataFiles
        self.files = DataFiles(self.root, metadata_patterns=METADATA_PATH_PATTERNS)

    def test_001_metadata_is_read_from_the_file_names(self):
        """Every measurement must at least say which experiment it belongs to."""
        self.files.initialize()
        df = self.files.dataframe

        self.assertGreater(len(df), 0)
        self.assertTrue(df.index.is_unique)
        self.assertIn('exp', df.columns)
        self.assertLess(df['exp'].isna().sum(), 10)

    def test_002_the_corrections_leave_a_usable_table(self):
        """The whole cleaning pipeline, as it is really used."""
        self.files.initialize()
        self.files.finalize([add_additional_experimental_info,
                             fix_acquisition_errors,
                             delete_test_data])
        df = self.files.dataframe

        # Nothing that was meant to be thrown away is still there
        for word in ('is_test', 'is_dark', 'is_white', 'is_adn',
                     'is_black', 'is_blanche', 'is_anneau', 'is_plus_tard'):
            self.assertEqual(int(df[word].sum()), 0, f"{word} should have been removed")

        self.assertTrue((df['modalite'] == 'raman').all())

        # Only the reference spectra, which belong to no mouse, may be left
        # without a dose. Anything more means the configuration has a hole.
        self.assertLess(df['dose'].isna().sum(), 50)

    def test_003_the_spectra_stack_into_one_matrix(self):
        """The measurements share one wavelength axis, so a matrix exists."""
        from ramandata import RamanData

        self.files.initialize()
        self.files.finalize([add_additional_experimental_info,
                             fix_acquisition_errors,
                             delete_test_data])

        raman = RamanData(self.files).initialize(verbose=False)
        counts = raman.report(verbose=False)

        self.assertEqual(counts['offered'], counts['kept'] + counts['rejected'])
        self.assertGreater(counts['kept'], 0)

        # A handful of laboratory notes saved as .txt, and the odd measurement
        # made with the other instrument, are expected. A lot more than that
        # would mean something changed in how the files are written.
        self.assertLess(counts['rejected'], 20)

        X, y = raman.training_set('dose')
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(y.dtype, np.float64)



if __name__ == "__main__":
    # unittest.main()

    from datafiles import DataFiles
    from ramandata import RamanData

    root = helper_find_root_directory()

    surya_files = DataFiles(root, metadata_patterns=METADATA_PATH_PATTERNS).initialize()
    surya_files.finalize([add_additional_experimental_info,
                         fix_acquisition_errors,
                         delete_test_data])

    raman = RamanData(surya_files).initialize(verbose=False)
    counts = raman.report()
