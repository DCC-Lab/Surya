import os
from pathlib import Path
import pandas as pd
import time
import unittest
from multiprocessing import Lock, Queue
from collections import deque
from threading import Thread
import unicodedata
import shutil
import tempfile


class DataFiles:
    local_root = Path(tempfile.gettempdir()) / Path("datafiles_cache")
    complete_marker = Path("local-copy-valid")

    def __init__(self, root = None, extensions = ['.txt'], methods = None):
        self.root = Path(root)
        self.extensions = extensions
        
        self.metadata_methods = methods if methods is not None else []

        self.data_files_paths = Queue()
        self._data_files_lock = Lock()
        self._properties = []
        self.dataframe = None

    @property
    def has_valid_local_copy(self):
        if (self.local_root / self.complete_marker).exists():
            return True

        return False

    def invalidate_local_copy(self):
        (self.local_root / self.complete_marker).unlink()

    def mark_local_copy_as_valid(self):
        with open(self.local_root / self.complete_marker,"w") as file:
            file.write("complete")

    def register_metadata_extraction_method(self, method):
        if method not in self.metadata_methods:
            self.metadata_methods.append(method)

    def initialize(self, methods = None, use_cache = True):
        if methods is not None:
            for method in methods:
                self.register_metadata_extraction_method(method)

        threads = []

        if self.has_valid_local_copy:
            print(f"Local copy will be used. If you see unexpected results, delete {self.local_root / self.complete_marker}")
        else:
            print(f"No valid copy available. Will attempt to copy locally to {self.local_root / self.complete_marker}")

        queue = deque()
        copy_queue = deque()
        threads.append(Thread(target=self.get_data_file_paths, args=( (queue, copy_queue), ) ))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, True)))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, False)))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, False)))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, False)))
        threads.append(Thread(target=self.copy_files_locally, args=(copy_queue, )))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        self.dataframe = pd.DataFrame(self._properties)
        self.dataframe.to_pickle("test.pkl")

    def finalize(self, methods):

        for method in methods:
            ret = method(self.dataframe)
            if isinstance(ret, pd.DataFrame):
                self.dataframe = ret
            else:
                raise ValueError("The finalize method() must return the final dataframe")

    def copy_files_locally(self, queue):
        if self.has_valid_local_copy:
            return

        next_time = time.time() + 2
        files = 0
        while True:
            try:
                element = queue.popleft()
            except IndexError:
                time.sleep(0.01)
                continue

            if element is not None:
                absolute_path, relative_path = element
            else:
                break

            dest_path = self.local_root / relative_path
            files +=  1
            if dest_path.exists() and dest_path.stat().st_size == absolute_path.stat().st_size:
                continue

            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(absolute_path, dest_path)             # copy2 preserve les dates

            if time.time() > next_time:
                next_time = time.time() + 2
                print(f"Copying {copies} so far")

        self.mark_local_copy_as_valid()

    def get_files_metadata(self, queue, progress):
        next_time = time.time() + 2

        root = self.root
        if self.has_valid_local_copy:
            root = self.local_root

        while True:
            try:
                element = queue.popleft()
            except IndexError:
                time.sleep(0.01)
                continue

            if element is not None:
                absolute_path, relative_path = element

                properties = {"file":relative_path}

                for method in self.metadata_methods:
                    properties.update(method(root, relative_path))


                with self._data_files_lock:
                    self._properties.append(properties)

                    if progress and time.time() > next_time:
                        print(f"Metadata from {len(self._properties)} files read")
                        next_time = time.time() + 2

            else:
                queue.appendleft(None) # Put back for other tasks
                break
    
    def get_single_file_metadata(self, root, file_relative_path):

        properties = {"file":file_relative_path}

        if self.has_valid_local_copy:
            local_copy_file_path = self.local_root / Path(file_relative_path)
            if local_copy_file_path.exists():
                root = self.local_root

        for method in self.metadata_methods:
            properties.update(method(root, file_relative_path))

        return properties


    def get_data_file_paths(self, queues, invisible_files=False, progress=False):
        """
        Get the list of files at a given root directory

        This could be a slow operation on the network.

        """
        

        if self.has_valid_local_copy:
            root = self.local_root
        else:
            root = self.root

        if not Path(root).exists():
            raise ValueError(f"The path {root} does not exist")

        all_files = []
        
        if not all_files:
            next_progress_print = time.time() + 2
            for dirpath, dirs, files in os.walk(root):
                for name in files:
                    path = unicodedata.normalize('NFC', os.path.join(dirpath, name))
                    if Path(path).suffix not in self.extensions:
                        continue

                    if "/." in path and not invisible_files:
                        continue

                    if progress and time.time() > next_progress_print:
                        print(".", end = "", flush=True)
                        next_progress_print = time.time() + 2

                    file_relative_path = str(Path(path).relative_to(root))

                    for queue in queues:
                        queue.append((Path(path), Path(file_relative_path)))

        for queue in queues:            
            queue.append(None)

        return all_files

    def validate_unique_metadata(self, ignore=(), verbose=True):
        """
        Verifie que les metadata de chaque fichier sont uniques.

        Pour chaque ligne, on rassemble les metadata dans un dictionnaire, on
        enleve les colonnes qui sont toujours differentes (time, indice1, file) puis
        on verifie que la signature qui reste n'apparait qu'une seule fois.

        Retourne un dictionnaire {signature: [liste des fichiers]} pour les
        signatures qui apparaissent plus d'une fois (donc les doublons).
        """

        df = self.dataframe

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
                print_debug(f"Metadata is unique ({len(df)} fichiers, colonnes: {colonnes})")
            else:
                print(f"{len(doublons)} signatures non-uniques touchant {n_doublons} fichiers:")
                for signature, indices in doublons.items():
                    if len(indices) % 5 != 0:
                        print(f"\n #{len(indices)} {dict(signature)}")
                        for i in indices:
                            print(f"    {df.loc[i, 'file']}")

        # On retourne les fichiers plutot que les indice1, plus utile pour le diagnostic
        return {sig: df.loc[idx, "file"].tolist() for sig, idx in doublons.items()}

from surya_experiments import *

class TestDataFiles(unittest.TestCase):
    def setUp(self):
        self.root = "/Volumes/Labdata/dcclab/surya" #helper_find_root_directory()
        if not Path(self.root).exists():
            self.root = "."

    def test_init(self):
        self.assertIsNotNone(DataFiles(self.root))

    def test_initialize(self):
        files = DataFiles(self.root, methods = [extract_properties_from_path, extract_header_from_relative_path])
        files.initialize()
        files.finalize([fix_acquisition_errors, add_additional_experimental_info, delete_test_data])
        files.validate_unique_metadata()

        mask = get_mask(files.dataframe, {'exp':2, "souris":27})
        print(files.dataframe[mask])


    def test_initialize_no_meta(self):
        files = DataFiles(self.root)
        files.initialize()

if __name__ == "__main__":
    unittest.main()





