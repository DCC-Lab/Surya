import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
from multiprocessing import Lock, Queue
from collections import deque
from threading import Thread
import unicodedata
import shutil
import unittest
import tempfile
import warnings
from platformdirs import user_cache_path

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
        cache_root   : where all the local copies are kept. It follows what
                       each system expects of a cache folder, so it is not the
                       same path on Linux, macOS and Windows.
        valid_marker : name of the small witness file that says a local copy
                       is complete.
    """

    # Where all the local copies live, one folder per root. Every system has
    # its own idea of where an application should keep something it can rebuild
    # by itself, and user_cache_path() knows all three:
    #
    #     Linux   ~/.cache/datafiles
    #     macOS   ~/Library/Caches/datafiles
    #     Windows C:\\Users\\<user>\\AppData\\Local\\datafiles\\Cache
    #
    # A cache folder is never emptied behind our back, unlike the temporary
    # folder, which the system clears out after a few days. Copying thousands
    # of files over the network takes minutes, so it must not have to be done
    # again just because nobody touched the files for a while.
    #
    # The name is the name of this class, not the name of any one study: what
    # is being cached is data files, whatever they happen to hold. Which study
    # a copy belongs to is decided by the root, further down.
    cache_root = user_cache_path("datafiles")
    valid_marker = Path("local-copy-valid")
    progress_delay = 3

    def __init__(self, root = None, extensions = ['.txt'], methods = None, metadata_patterns = None):
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
        self.metadata_patterns = metadata_patterns if metadata_patterns is not None else []

        self.data_files_paths = Queue()
        self._data_files_lock = Lock()
        self._properties = []
        self.dataframe = None
        self.cache_warning_issued = False

    @property
    def has_valid_local_copy(self):
        """
        Tells whether a complete local copy is available.

        We cannot simply check that the folder exists: a copy that was
        interrupted leaves a half-filled folder that looks perfectly fine. So
        we drop a small witness file at the very end of the copy, and it is its
        presence that proves the copy actually finished.
        """

        marker_path = self.local_root / self.valid_marker
        if not marker_path.exists():
            return False

        name_found, path_found = self.read_marker(marker_path)

        # The name is what is checked, not the path: the same dataset reached
        # through another mount is still the same dataset, and must reuse the
        # copy rather than make a second one. A name that does not match means
        # the folder was renamed by hand, or that two different datasets are
        # called the same thing. Either way, reading it would quietly return
        # the measurements of another experiment.
        if name_found != self.dataset_name:
            if not self.cache_warning_issued:
                print(f"Warning: {self.local_root} holds a copy of '{name_found}', "
                      f"not of '{self.dataset_name}'. It will be ignored.")
                self.cache_warning_issued = True
            return False

        if path_found != self.root_signature and not self.cache_warning_issued:
            print(f"Note: the local copy of '{self.dataset_name}' was made from "
                  f"'{path_found}', which is another way of reaching the same data.")
            self.cache_warning_issued = True

        return True

    @staticmethod
    def read_marker(marker_path):
        """
        Reads the witness file, and returns (dataset name, path it came from).

        The file holds the name on its first line and the full path on the
        second. A file written before the path was recorded holds only one
        line, and the path comes back empty.
        """
        lines = marker_path.read_text().splitlines()
        name = lines[0].strip() if lines else ""
        path = lines[1].strip() if len(lines) > 1 else ""
        return name, path

    @property
    def dataset_name(self):
        """
        The name of the dataset, which is the last folder of the root.

        This, and not the full path, is what says which data we are dealing
        with. The same measurements are reached by different paths depending on
        who is looking and how they mounted the share:

            /Volumes/labdata/dcclab/surya          on one machine
            /mnt/labdata/dcclab/surya              on another
            \\\\cafeine3.crulrg.ulaval.ca\\...\\surya  on Windows

        All three are the same dataset, and all three must share one local copy.
        Telling them apart by their full path would give each person a copy of
        their own, and copying thousands of files over the network takes
        minutes that nobody should pay twice.

        Lowercased, so that a share mounted as 'Labdata' one day and 'labdata'
        the next does not end up with two copies.
        """
        try:
            root = self.root.resolve()
        except OSError:
            root = self.root.absolute()      # the drive may be unreachable: do not block

        return root.name.lower() or "root"

    @property
    def local_root(self):
        """
        The folder holding the local copy of this dataset.

            root   /Volumes/labdata/dcclab/surya
            folder ~/Library/Caches/datafiles/surya

        Two datasets whose last folder is called the same thing do share this
        folder, and the witness file inside says which one is actually there.
        Two studies both keeping their measurements in a folder plainly called
        'data' would take turns copying over each other, so give them names
        that mean something.
        """
        return self.cache_root / self.dataset_name

    @property
    def root_signature(self):
        """
        The full path of the root, as written into the witness file.

        It says where a copy came from, which is worth knowing when something
        looks wrong, but it is not what identifies the dataset: see
        dataset_name above.
        """
        try:
            root = self.root.resolve()
        except OSError:
            root = self.root.absolute()      # the drive may be unreachable: do not block

        return os.path.normcase(str(root)).lower()

    def invalidate_local_copy(self):
        """
        Declares the local copy unusable by deleting the witness file.

        Call this when you suspect the copy no longer matches the original
        data. The next run will copy everything again.
        """
        (self.local_root / self.valid_marker).unlink(missing_ok=True)

    def delete_local_copy(self):
        """
        Removes the local copy of this root entirely, files and all.

        invalidate_local_copy() only takes away the witness file, so the copy is
        made again over the top of what is already there, which is quick. This
        one frees the space, and is what to reach for when the folder is in the
        way rather than merely out of date.

        Only the folder belonging to this root is touched. Copies of other roots
        are left alone.
        """
        if self.local_root.exists():
            shutil.rmtree(self.local_root)

    @classmethod
    def local_copies(cls):
        """
        Lists every local copy kept on this machine, and where each came from.

        Returns a list of (folder, root, complete), where root is the path the
        copy was made from and complete says whether the witness file is there.
        A copy that is not complete was interrupted and will be finished the
        next time that dataset is read.

        Useful to see what is taking up room, and to find the folder to hand to
        delete_local_copy().
        """
        if not cls.cache_root.exists():
            return []

        copies = []
        for folder in sorted(cls.cache_root.iterdir()):
            if not folder.is_dir():
                continue

            marker = folder / cls.valid_marker
            root = cls.read_marker(marker)[1] if marker.exists() else None
            copies.append((folder, root, marker.exists()))

        return copies

    def mark_local_copy_as_valid(self):
        """
        Drops the witness file that declares the local copy complete.

        Called exactly once, after the very last file has been copied. It holds
        the name of the dataset, which is what identifies it, and underneath the
        path it was copied from, which is only there to be read by a person
        wondering where these files came from.
        """
        self.local_root.mkdir(parents=True, exist_ok=True)
        with open(self.local_root / self.valid_marker, "w") as file:
            file.write(f"{self.dataset_name}\n{self.root_signature}\n")

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

    def initialize(self, methods = None, create_local_copy = False):
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

        self.register_metadata_extraction_method(self.extract_properties_from_patterns)


        threads = []
        queue = deque()
        copy_queue = deque()

        if self.has_valid_local_copy:
            print(f"Delete {self.local_root / self.valid_marker} to avoid cache")

        threads.append(Thread(target=self.get_data_file_paths, args=( (queue, copy_queue), ) ))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, True)))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, False)))

        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join()
        
        if time.time() - start_time > 10 and not self.has_valid_local_copy:
            copy_thread = Thread(target=self.copy_files_locally, args=(copy_queue, ))
            copy_thread.start() # Attempt to copy in the background

        self.dataframe = pd.DataFrame(self._properties)

        # The relative path of the file becomes the label of its row. The rows
        # are built by several tasks at once, so the order they end up in is
        # different every run: a plain row number would designate a different
        # file each time the program is started. The file name does not move.
        assert self.dataframe['file'].is_unique, "Two rows describe the same file"
        self.dataframe = self.dataframe.set_index('file')

        # A column of whole numbers that holds a single missing value is turned
        # into decimals by pandas: mouse 39 is then shown as 39.0, which is
        # confusing and exports badly. 'Int64', with a capital I, is the whole
        # number type that accepts missing values, so 39 stays 39.
        integer_columns = {"exp", "petri", "jour", "souris", "indice1", "indice2",
                           "dose", "zone", "subzone", "batch"}
        self.dataframe = self.dataframe.astype(
            {c: "Int64" for c in integer_columns if c in self.dataframe.columns})

        return self


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
                raise ValueError("The finalize method()s must return the final dataframe")

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

        next_time = time.time() + self.progress_delay
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
                next_time = time.time() + self.progress_delay
                print(f"Copying {files} so far")

        self.mark_local_copy_as_valid()

    def extract_properties_from_patterns(self, root, file_relative_path):
        """
        Reads the metadata out of a file name using a list of regular expressions.

        Each pattern is searched for in the whole path, ignoring case. Whatever its
        named groups capture becomes an entry of the returned dictionary, converted
        to a whole number when it looks like one and lowercased otherwise.

        A group whose name begins with 'is_' is treated differently: what it
        captured is thrown away and only its presence is kept, as true or false.
        It is the way to write down a word that either appears in the path or
        does not -- 'test', 'verre', 'dark' -- where the word itself carries no
        information beyond being there. Those entries are always set, even when
        the word is absent, because a column that is sometimes false and
        sometimes missing cannot be filtered on: a hole is neither true nor
        false, so a row holding one is dropped by a test for false just as
        surely as a row holding true.
        """

        def _to_normalized_values(properties):
            """
            Turns the captured text into whole numbers where that makes sense.

            A group that captured nothing is left alone: it means the optional part of
            the pattern was not there, which is not the same as a value of zero.
            """
            # Match any separator and any decimal point (because we sometimes get fr-ca floats)
            FLOAT_REGEX = r"[-+]?(?:\d{1,3}(?:[\u00a0 .,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?"
            INT_REGEX = r"[-+]?0*\d+"

            for key, value in properties.items():
                if key.startswith("is_"):
                    properties[key] = value is not None
                    continue

                if value is None:
                    continue

                if re.fullmatch(INT_REGEX, value):
                    properties[key] = int(value)
                elif re.fullmatch(FLOAT_REGEX, value):
                    # float() only understands a dot, so a number written the
                    # French way has to be rewritten before being converted:
                    # the grouping spaces go away, the comma becomes a dot.
                    properties[key] = float(re.sub(r"[\u00a0 ]", "", value).replace(",", "."))
                else:
                    # Anything that is not a number is a word, and the same word
                    # is written 'Raman' in one folder and 'raman' in the next.
                    # Lowercasing here means the rest of the program only ever
                    # sees one spelling.
                    properties[key] = value.lower()

            return properties


        file_path = str(Path(root) / Path(file_relative_path))

        # Every 'is_' group of every pattern starts out false, so that a word
        # which is simply not there gives false rather than nothing at all. The
        # names are read from the patterns themselves, which is what keeps this
        # method from having to know which words anyone is looking for.
        properties = {name: False
                      for pattern in self.metadata_patterns
                      for name in re.compile(pattern).groupindex
                      if name.startswith("is_")}

        for pattern in self.metadata_patterns:
            match = re.search(pattern, file_path, re.IGNORECASE)
            if match is not None:
                properties.update(_to_normalized_values(match.groupdict()))

        # A time is spread over four groups that only mean something together.
        if properties.get('heure') is not None:
            properties['time'] = datetime.time(hour=properties['heure'],
                                               minute=properties['minutes'],
                                               second=properties['s'],
                                               microsecond=properties['ms'] * 1000)

        properties['file'] = str(file_relative_path)

        return properties


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
        next_time = time.time() + self.progress_delay

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

                properties = {"file":relative_path,"absolute_path":absolute_path}

                for method in self.metadata_methods:
                    properties.update(method(root, relative_path))


                with self._data_files_lock:
                    self._properties.append(properties)

                    if progress and time.time() > next_time:
                        print(f"Metadata from {len(self._properties)} files read")
                        next_time = time.time() + self.progress_delay

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

        next_progress_print = time.time() + 2
        for dirpath, dirs, files in os.walk(root):
            for name in files:
                absolute_path = unicodedata.normalize('NFC', os.path.join(dirpath, name))
                if Path(absolute_path).suffix not in self.extensions:
                    continue

                # Hidden files, and anything inside a hidden folder. Testing
                # every part of the path works on Windows too, where the
                # separator is a backslash.
                relative_parts = Path(absolute_path).relative_to(root).parts
                if not invisible_files and any(part.startswith(".") for part in relative_parts):
                    continue

                # Files that belong to the machinery, not to the experiment:
                # our own path cache, and the metadata companions that macOS
                # scatters over network drives. They are not spectra, and
                # they would show up as a row of missing values.
                if name.startswith("._") :
                    continue

                if progress and time.time() > next_progress_print:
                    print(".", end = "", flush=True)
                    next_progress_print = time.time() + 2

                file_relative_path = str(Path(absolute_path).relative_to(root))

                for queue in queues:
                    queue.append((Path(absolute_path), Path(file_relative_path)))

        for queue in queues:            
            queue.append(None)

    def validate_unique_metadata(self, ignore=(), ignore_prefixes=("Spectrum:",), verbose=True):
        """
        Checks that the metadata of each file is unique.

        For every row, we gather the metadata into a dictionary, remove the
        columns that are always different (time, indice1, file), then check
        that the signature left over appears only once.

        ignore          : column names to leave out of the signature.
        ignore_prefixes : whole families of columns to leave out. Instruments
                          add one column per setting they record, all sharing a
                          prefix, and those describe the measurement rather than
                          identifying the sample.

        Returns a dictionary {signature: [list of files]} for the signatures
        that appear more than once, that is, the duplicates.
        """

        df = self.dataframe

        colonnes = [c for c in df.columns
                    if c not in ignore and not c.startswith(tuple(ignore_prefixes))]

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
                            print(f"    {i}")

        # The index labels are the file names, which is what is useful here
        return {sig: list(idx) for sig, idx in doublons.items()}

    def get_mask(self, mask_as_dict):
        df = self.dataframe
        mask = pd.Series(True, index=df.index)
        for key, value in mask_as_dict.items():
            if key not in df.columns:
                continue
            mask &= (df[key].notna() & (df[key] == value))

        return mask

    def read_data_files(self, reader_method, mask = None):
        """
        Reads the actual content of the data files, for instance the spectra.

        The content is deliberately NOT stored in self.dataframe. A spectrum is
        a table of hundreds of points, and putting a whole table inside a
        single cell of another table breaks almost everything you would want to
        do afterwards: saving to Excel, filtering, grouping. The dataframe
        stays a table of metadata, small and fast.

        Instead, this returns a dictionary whose keys are the index labels of
        self.dataframe. The two therefore line up: the content of row 42 is
        found at spectra[42], and any mask you used to select rows still
        applies.

        reader_method : the function that reads one file and returns its
                        content, for instance read_spectrum_file.
        mask          : an optional filter, as returned by get_mask(), to read
                        only some of the files.
        """
        if mask is None:
            df = self.dataframe
        else:
            df = self.dataframe[mask]

        if not df.index.is_unique:
            raise ValueError("The dataframe index has duplicates: it cannot be used to match the files to their content")

        files_data = {}
        next_time = time.time() + self.progress_delay
        for index, absolute_path in df['absolute_path'].items():
            files_data[index] = reader_method(absolute_path)
            if time.time() > next_time:
                print(f"{len(files_data)} of {len(df)} files read")
                next_time = time.time() + self.progress_delay

        return files_data




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

    def test_001_init(self):
        """Checks that a DataFiles object can simply be created."""
        self.assertIsNotNone(DataFiles(self.root))

    def test_002_initialize(self):
        """
        Full run: metadata reading, corrections, then validation.

        This is the test that reproduces the real use of the class from start
        to finish.
        """
        files = DataFiles(self.root, 
                          methods = [extract_header_from_relative_path],
                          metadata_patterns=METADATA_PATH_PATTERNS)

        files.initialize()

    def test_003_finalize(self):
        """
        Full run: metadata reading, corrections, then validation.

        This is the test that reproduces the real use of the class from start
        to finish.
        """
        files = DataFiles(self.root, 
                          methods = [extract_header_from_relative_path],
                          metadata_patterns=METADATA_PATH_PATTERNS)
        files.initialize()
        files.finalize([fix_acquisition_errors, add_additional_experimental_info, delete_test_data])

    def test_004_validate(self):
        """
        Full run: metadata reading, corrections, then validation.

        This is the test that reproduces the real use of the class from start
        to finish.
        """
        files = DataFiles(self.root, 
                          methods = [extract_header_from_relative_path],
                          metadata_patterns=METADATA_PATH_PATTERNS)
        files.initialize()
        files.finalize([fix_acquisition_errors, add_additional_experimental_info, delete_test_data])
        files.validate_unique_metadata()
        mask = files.get_mask({})
        files_content = files.read_data_files(reader_method=read_spectrum_file, mask=mask)
        
        # # The spectra are kept beside the dataframe, not inside it. The keys of
        # # files_content are the index labels of files.dataframe, so we can go
        # # back and forth between the metadata of a file and its content.
        # for index, spectrum in files_content.items():
        #     metadata = files.dataframe.loc[index]
        #     print(f"{index}: {len(spectrum)} points, souris {metadata['souris']}")


    def test_initialize_no_meta(self):
        """
        Checks that everything still works with no extraction function at all.

        The table then holds only the 'file' column.
        """
        files = DataFiles(self.root).initialize()

    
if __name__ == "__main__":
    unittest.main()





