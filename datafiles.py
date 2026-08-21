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
    """
    Collects the metadata of every data file in a directory into one table.

    You give it a starting directory (`root`) and a few functions that know how
    to read information out of a file (mouse number, day, spectrometer
    settings, and so on). The class walks through all the files, applies those
    functions to each one, and gathers everything into a pandas DataFrame,
    which is a table much like an Excel sheet.

    The data usually lives on a network drive, which is very slow: it takes
    about 7 milliseconds just to open one file, compared to 0.02 millisecond on
    the local disk. So the class does two things at the same time: it reads the
    metadata, and it copies the files to the local disk so that the next runs
    are fast.

    Typical use:

        files = DataFiles("/Volumes/labdata/dcclab/surya",
                          methods=[extract_properties_from_path])
        files.initialize()
        files.finalize([some_methods, ...])
        print(files.dataframe)

    Class attributes:
        local_root      : where the local copy of the files is kept.
        complete_marker : name of the small witness file that says the local
                          copy is complete.
    """

    local_root = Path(tempfile.gettempdir()) / Path("datafiles_cache")
    complete_marker = Path("local-copy-valid")

    def __init__(self, root = None, extensions = ['.txt'], methods = None):
        """
        Sets up the object without reading anything from disk yet.

        root       : the directory where the data files should be looked for.
        extensions : the kinds of files we care about, for instance ['.txt'].
                     Every other kind is ignored.
        methods    : the functions that know how to extract metadata. Each one
                     receives (root, relative_path) and returns a dictionary.

        The real work only starts when you call initialize().
        """
        self.root = Path(root)
        self.extensions = extensions
        
        self.metadata_methods = methods if methods is not None else []

        self.data_files_paths = Queue()
        self._data_files_lock = Lock()
        self._properties = []
        self.dataframe = None

    @property
    def has_valid_local_copy(self):
        """
        Tells whether a complete local copy is available.

        We cannot simply check that the folder exists: a copy that was
        interrupted leaves a half-filled folder that looks perfectly fine. So
        we drop a small witness file at the very end of the copy, and it is its
        presence that proves the copy actually finished.
        """
        if (self.local_root / self.complete_marker).exists():
            return True

        return False

    def invalidate_local_copy(self):
        """
        Declares the local copy unusable by deleting the witness file.

        Call this when you suspect the copy no longer matches the original
        data. The next run will copy everything again.
        """
        (self.local_root / self.complete_marker).unlink()

    def mark_local_copy_as_valid(self):
        """
        Drops the witness file that declares the local copy complete.

        Called exactly once, after the very last file has been copied.
        """
        with open(self.local_root / self.complete_marker,"w") as file:
            file.write("complete")

    def register_metadata_extraction_method(self, method):
        """
        Adds one metadata extraction function to the list.

        The function receives (root, relative_path) and must return a
        dictionary, for example {'souris': 3, 'jour': 8}. You may register
        several of them: their results are merged together for each file. The
        same function is never added twice.
        """
        if method not in self.metadata_methods:
            self.metadata_methods.append(method)

    def initialize(self, methods = None, use_cache = True):
        """
        Does all the work: walks the files and fills self.dataframe.

        Six tasks are started side by side. A task (a "thread") is a line of
        execution that moves forward at the same time as the others:

          - 1 task walks the directories and announces the files it finds by
            dropping them into two waiting lines (queues);
          - 4 tasks read the metadata of the announced files;
          - 1 task copies the files to the local disk.

        Why several tasks? Because the computer spends most of its time waiting
        for the network. While one task waits for an answer, the others work.

        The method only returns once everything is finished: that is what the
        join() calls do, they wait for each task to end.
        """
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
        """
        Applies a series of corrections to the table once it is built.

        Each method receives the whole DataFrame and must return a new one:
        corrected, filtered or enriched. This is where everything that needs a
        global view goes, that is, everything that cannot be done one file at a
        time.

        Raises an error if a method forgets to return the table.
        """
        for method in methods:
            ret = method(self.dataframe)
            if isinstance(ret, pd.DataFrame):
                self.dataframe = ret
            else:
                raise ValueError("The finalize method() must return the final dataframe")

    def copy_files_locally(self, queue):
        """
        Copies to the local disk the files announced in the waiting line.

        This method runs in its own task. It takes files out of `queue` one by
        one until it finds None, which is the agreed signal meaning "there will
        be no more".

        A file already present locally with the same size is not copied again,
        so you can restart the program without redoing everything.

        If a valid copy already exists, the method steps aside immediately.
        """
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

            shutil.copy2(absolute_path, dest_path)             # copy2 preserves the modification dates

            if time.time() > next_time:
                next_time = time.time() + 2
                print(f"Copying {copies} so far")

        self.mark_local_copy_as_valid()

    def get_files_metadata(self, queue, progress):
        """
        Reads the metadata of the files announced in the waiting line.

        This method runs in its own task, and four copies of it run at the same
        time. Each one takes a file out of `queue`, applies every registered
        function to it, and stores the resulting dictionary in
        self._properties.

        The `_data_files_lock` lock prevents two tasks from writing into that
        list at exactly the same instant, which could corrupt it.

        The value None in the queue means "finished". We put it back before
        leaving, so that the three other tasks see it too.

        progress : only one of the four tasks receives True, otherwise the
                   progress message would be printed four times over.
        """
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
    
    def get_data_file_paths(self, queues, invisible_files=False, progress=False):
        """
        Walks the directories and announces every data file it finds.

        This is the only task that explores the disk, and it can be a slow
        operation over the network. For each file it keeps, it drops the pair
        (full path, relative path) into ALL the waiting lines it was given: one
        for reading the metadata, one for the local copy.

        At the end it drops None into each queue: that is the agreed signal
        telling the other tasks there will be nothing more to process.

        The names are normalized to NFC because macOS writes accented
        characters in a way (the letter and the accent stored separately) that
        other tools do not always recognize.

        queues          : the waiting lines to feed.
        invisible_files : whether to include hidden files (those starting
                          with a dot).
        progress        : whether to print a dot every two seconds.
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
        Checks that the metadata of each file is unique.

        For every row, we gather the metadata into a dictionary, remove the
        columns that are always different (time, indice1, file), then check
        that the signature left over appears only once.

        Returns a dictionary {signature: [list of files]} for the signatures
        that appear more than once, that is, the duplicates.
        """

        df = self.dataframe

        colonnes = [c for c in df.columns if c not in ignore and not c.startswith("Spectrum:")]

        signatures = {}
        for i, row in df[colonnes].iterrows():
            # The metadata of this row, without the missing values
            metadata = {k: v for k, v in row.items() if pd.notna(v)}
            # A dict is not hashable: we turn it into a sorted tuple for the key
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

        # We return the files rather than the indice1, more useful for diagnostics
        return {sig: df.loc[idx, "file"].tolist() for sig, idx in doublons.items()}

from surya_experiments import *

class TestDataFiles(unittest.TestCase):
    """
    Automated tests for the DataFiles class.

    You run them by executing this file directly. unittest calls setUp() before
    each test, then every method whose name starts with "test".
    """

    def setUp(self):
        """
        Picks the data directory before each test.

        If the network drive is not mounted, we fall back to the current
        directory so that the tests can still run.
        """
        self.root = "/Volumes/Labdata/dcclab/surya" #helper_find_root_directory()
        if not Path(self.root).exists():
            self.root = "."

    def test_init(self):
        """Checks that a DataFiles object can simply be created."""
        self.assertIsNotNone(DataFiles(self.root))

    def test_initialize(self):
        """
        Full run: metadata reading, corrections, then validation.

        This is the test that reproduces the real use of the class from start
        to finish.
        """
        files = DataFiles(self.root, methods = [extract_properties_from_path, extract_header_from_relative_path])
        files.initialize()
        files.finalize([fix_acquisition_errors, add_additional_experimental_info, delete_test_data])
        files.validate_unique_metadata()

    def test_initialize_no_meta(self):
        """
        Checks that everything still works with no extraction function at all.

        The table then holds only the 'file' column.
        """
        files = DataFiles(self.root)
        files.initialize()

if __name__ == "__main__":
    unittest.main()





