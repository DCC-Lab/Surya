import os
import re
import datetime
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
    r"""
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

        files = DataFiles("/Volumes/share/measurements",
                          metadata_patterns=[r"sample(?P<sample>\d+)"])
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
                     Two methods are already always included: 
                        - extract_properties_from_patterns, using named regex (see the function)
                        - extract_extended_properties_from_path, file size, etc..

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
        # What the background tasks ran into. Each of them catches its own
        # errors, because an error raised in one task is seen by nobody, and
        # puts them here for initialize() to raise once they have all stopped.
        self.errors = []

    @property
    def has_valid_local_copy(self):
        """
        Tells whether a complete local copy is available.

        We cannot simply check that the folder exists: a copy that was
        interrupted leaves a half-filled folder that looks perfectly fine. So
        we drop a small witness file at the very end of the copy, and it is its
        presence that proves the copy actually finished.
        """

        with self._data_files_lock:
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

            /Volumes/share/measurements        on one machine
            /mnt/share/measurements            on another
            \\\\server.example.com\\...\\measurements  on Windows

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

            root   /Volumes/share/measurements
            folder ~/Library/Caches/datafiles/measurements

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
        dictionary, for example {'sample': 3, 'day': 8}. You may register
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
        self.register_metadata_extraction_method(self.extract_extended_properties_from_path)

        threads = []
        queue = deque()
        copy_queue = deque()

        # if self.has_valid_local_copy:
        #     print(f"Delete {self.local_root / self.valid_marker} to avoid cache")

        threads.append(Thread(target=self.get_data_file_paths, args=( (queue, copy_queue), ) ))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, True)))
        threads.append(Thread(target=self.get_files_metadata, args=(queue, False)))

        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # A task that died took its error down with it: join() comes back as if
        # all were well, and the failure would only show up much later as a
        # missing column or an empty table. Raising it here says what actually
        # went wrong, where it went wrong.
        if self.errors:
            raise self.errors[0]

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
        # into decimals by pandas: sample 39 is then shown as 39.0, which is
        # confusing and exports badly. 'Int64', with a capital I, is the whole
        # number type that accepts missing values, so 39 stays 39.
        #
        # Which columns those are is decided by looking at what they hold, not
        # by a list of names: this class has no idea what anyone is measuring.
        # A column holding a real decimal anywhere is left alone, because
        # turning 2.5 into a whole number would either fail or lose the half.
        for column in self.dataframe.columns:
            values = self.dataframe[column].dropna()

            # True and False count as numbers to pandas, and 'is_something'
            # columns must stay true or false rather than become 1 and 0.
            if pd.api.types.is_bool_dtype(values):
                continue
            if values.empty or not pd.api.types.is_numeric_dtype(values):
                continue
            if (values == values.round()).all():
                self.dataframe[column] = self.dataframe[column].astype("Int64")

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

        try:
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

                shutil.copy2(absolute_path, dest_path)         # copy2 preserves the modification dates

                if time.time() > next_time:
                    next_time = time.time() + self.progress_delay
                    print(f"Copying {files} so far")

            # Only once every file went through, and only if nothing went
            # wrong: a copy that gave up halfway must not be declared complete.
            self.mark_local_copy_as_valid()

        except Exception as error:
            # This task runs on its own, so raising here would tell nobody.
            # The copy stays unmarked, which is what we want: it will simply be
            # made again next time rather than being trusted as it stands.
            self.errors.append(error)

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

    def extract_extended_properties_from_path(self, root, file_relative_path):
        """
        Get more information about the file_path, but slow.
        Returns a dictionary with the properties
        """
        extended_properties = {}

        file_path = str(Path(root) / Path(file_relative_path))

        # Fetch file_path stats (slow)
        try:
            file_info = Path(file_path).stat()
            
            extended_properties['size_in_bytes'] = file_info.st_size
            # Others possible
        except Exception as e:
            pass # We just give up if unable to do it

        return extended_properties


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

        try:
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

        except Exception as error:
            # This task runs on its own, so raising here would tell nobody.
            # The error is put aside for initialize() to raise afterwards, and
            # the None is put back so that the other readers still stop.
            self.errors.append(error)
            queue.append(None)

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

        Whatever happens, the None is dropped: the try below makes sure of it.
        This task runs on its own, so an error here would otherwise be seen by
        nobody, and the tasks reading from the queues would wait for a signal
        that never comes. The error is put aside instead, for initialize() to
        raise once every task has stopped.

        queues          : the waiting lines to feed.
        invisible_files : whether to include hidden files (those starting
                          with a dot).
        progress        : whether to print a dot every two seconds.
        """
        try:
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

        except Exception as error:
            self.errors.append(error)

        finally:
            # In the finally, not after the loop: the tasks waiting on these
            # queues must be released even when the walk gave up halfway.
            for queue in queues:
                queue.append(None)

    def validate_unique_metadata(self, ignore=("absolute_path",),
                                 ignore_prefixes=("Spectrum:",), verbose=True):
        """
        Checks that the metadata of each file is unique.

        For every row, we gather the metadata into a dictionary, remove the
        columns that are always different, then check that the signature left
        over appears only once. Two files carrying exactly the same metadata
        cannot be told apart afterwards, which usually means one acquisition
        was named after another by mistake.

        ignore          : column names to leave out of the signature.
                          'absolute_path' is left out by default: it is not
                          metadata, it is where the file happens to sit on this
                          machine, and it differs for every row, so leaving it
                          in would hide every duplicate there is.
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
        stays a table of metadata, small and fast. If the data file are images,
        it is even worse.

        Instead, this returns a dictionary whose keys are the index labels of
        self.dataframe. The two therefore line up: the content of file 'file' is
        found at spectra['file'], and any mask you used to select rows still
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




def write_data_file(path, lines=("1.0 10.0", "2.0 20.0")):
    """
    Writes a small file for the tests to find.

    The content does not matter to DataFiles, which never looks inside a file:
    it walks folders, reads names, and hands the files over to whoever knows
    what they hold. Two lines of numbers are enough to make the file real.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestDataFiles(unittest.TestCase):
    """
    Tests for DataFiles, run against files written by the tests themselves.

    Nothing here belongs to any particular study. DataFiles walks folders,
    reads what the names say and keeps a local copy; it does not know whether
    the files hold spectra, images or anything else. So the tests describe
    made-up measurements, with made-up metadata, in a temporary folder.

    That also makes them fast and repeatable, and lets them arrange on purpose
    the awkward situations that are rare in real life: a folder that cannot be
    read, a copy interrupted halfway, two files that say exactly the same thing.

    The cache is redirected into that same temporary folder. Without this, a
    copy left over from real use would be read instead of the files written
    here, and the tests would quietly check the wrong data.
    """

    # Made-up metadata: a sample number, a dose, a zone, and two words that are
    # either there or not. Enough to exercise every kind of group.
    PATTERNS = [
        r"sample(?P<sample>\d+)",
        r"dose(?P<dose>[\d.,]+)",
        r"zone(?P<zone>\d+)",
        # The number that tells apart the repeated measurements of one zone.
        # Without it every repeat would carry exactly the same metadata, and
        # validate_unique_metadata() would rightly call them all duplicates.
        r"_(?P<number>\d+)\.txt",
        r"(?P<mode>alpha|beta)",
        r"(?P<is_test>tests?)",
        r"(?P<is_reference>reference)",
    ]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "data"

        # Two samples, two zones each, three files per zone.
        self.expected_files = 0
        for sample in (1, 2):
            for zone in (1, 2):
                folder = self.root / "alpha" / f"sample{sample}" / f"zone{zone}"
                for number in range(3):
                    write_data_file(folder / f"sample{sample}_dose45_zone{zone}_{number}.txt")
                    self.expected_files += 1

        # The cache is shared by every instance, so it is moved aside for the
        # duration of the test.
        self.saved_cache_root = DataFiles.cache_root
        DataFiles.cache_root = Path(self.temporary.name) / "cache"

    def tearDown(self):
        DataFiles.cache_root = self.saved_cache_root
        self.temporary.cleanup()

    def made(self, **kwargs):
        """A DataFiles pointing at the files written by setUp()."""
        kwargs.setdefault("metadata_patterns", self.PATTERNS)
        return DataFiles(self.root, **kwargs)

    # ---- creating -------------------------------------------------------

    def test_001_init(self):
        self.assertIsNotNone(self.made())

    def test_002_nothing_is_read_before_initialize(self):
        self.assertIsNone(self.made().dataframe)

    # ---- walking the folders --------------------------------------------

    def test_010_every_file_is_found(self):
        files = self.made().initialize()
        self.assertEqual(len(files.dataframe), self.expected_files)

    def test_011_the_file_name_is_the_row_label(self):
        """
        The rows are labelled by the file, not by a number.

        Several tasks fill the table at once, so the order changes from one run
        to the next: a row number would designate a different file every time.
        """
        files = self.made().initialize()

        self.assertEqual(files.dataframe.index.name, 'file')
        self.assertTrue(files.dataframe.index.is_unique)
        self.assertIn('sample1/zone1/sample1_dose45_zone1_0.txt',
                      {str(Path(i).relative_to('alpha')) for i in files.dataframe.index})

    def test_012_only_the_wanted_extensions(self):
        write_data_file(self.root / "alpha" / "notes.md")
        write_data_file(self.root / "alpha" / "table.csv")

        files = self.made().initialize()
        self.assertEqual(len(files.dataframe), self.expected_files)

    def test_013_several_extensions_at_once(self):
        write_data_file(self.root / "alpha" / "table.csv")

        files = self.made(extensions=['.txt', '.csv']).initialize()
        self.assertEqual(len(files.dataframe), self.expected_files + 1)

    def test_014_hidden_files_and_folders_are_left_out(self):
        write_data_file(self.root / "alpha" / ".hidden.txt")
        write_data_file(self.root / ".hidden_folder" / "inside.txt")

        files = self.made().initialize()
        self.assertEqual(len(files.dataframe), self.expected_files)

    def test_015_a_root_that_does_not_exist(self):
        missing = DataFiles(Path(self.temporary.name) / "nowhere")
        with self.assertRaises(Exception):
            missing.initialize()

    # ---- reading the names -----------------------------------------------

    def test_020_patterns_fill_the_columns(self):
        files = self.made().initialize()
        row = files.dataframe.iloc[0]

        self.assertIn(row['sample'], (1, 2))
        self.assertEqual(row['dose'], 45)
        self.assertEqual(row['mode'], 'alpha')

    def test_021_a_whole_number_stays_a_whole_number(self):
        """45 must not become 45.0: it exports badly and reads worse."""
        files = self.made().initialize()
        self.assertIsInstance(files.dataframe.iloc[0]['sample'], (int, np.integer))

    def test_022_leading_zeros_are_still_numbers(self):
        write_data_file(self.root / "alpha" / "sample007_dose45_zone1_0.txt")

        files = self.made().initialize()
        samples = set(files.dataframe['sample'].dropna())
        self.assertIn(7, samples)

    def test_023_a_decimal_written_the_french_way(self):
        """Computers set to French write 2,5 where others write 2.5."""
        write_data_file(self.root / "alpha" / "sample3_dose2,5_zone1_0.txt")

        files = self.made().initialize()
        doses = set(files.dataframe['dose'].dropna())
        self.assertIn(2.5, doses)

    def test_024_words_are_lowercased(self):
        """The same word is spelled two ways across folders, never three."""
        write_data_file(self.root / "BETA" / "sample9_dose45_zone1_0.txt")

        files = self.made().initialize()
        self.assertEqual(set(files.dataframe['mode'].dropna()), {'alpha', 'beta'})

    def test_025_presence_groups_are_true_or_false(self):
        write_data_file(self.root / "alpha" / "test" / "sample8_dose45_zone1_0.txt")

        files = self.made().initialize()
        df = files.dataframe

        self.assertEqual(df['is_test'].dtype, bool)
        self.assertEqual(int(df['is_test'].sum()), 1)

    def test_026_a_presence_group_is_never_missing(self):
        """
        A column that is sometimes false and sometimes missing cannot be
        filtered on: a hole is neither true nor false, so a row holding one is
        dropped by a test for false just as surely as a row holding true.
        """
        files = self.made().initialize()
        df = files.dataframe

        self.assertEqual(int(df['is_reference'].isna().sum()), 0)
        self.assertEqual(int(df['is_reference'].sum()), 0)

    def test_027_extraction_methods_are_merged_in(self):
        def extra(root, relative_path):
            return {'extra': 'yes'}

        files = self.made(methods=[extra]).initialize()
        self.assertTrue((files.dataframe['extra'] == 'yes').all())

    def test_028_a_method_is_never_registered_twice(self):
        def extra(root, relative_path):
            return {}

        files = self.made(methods=[extra])
        files.register_metadata_extraction_method(extra)
        self.assertEqual(len(files.metadata_methods), 1)

    def test_029_no_metadata_at_all(self):
        """With nothing to look for, the table still lists the files."""
        files = DataFiles(self.root).initialize()
        self.assertEqual(len(files.dataframe), self.expected_files)

    # ---- filtering and checking -------------------------------------------

    def test_030_get_mask(self):
        files = self.made().initialize()
        mask = files.get_mask({'sample': 1})

        self.assertEqual(int(mask.sum()), 6)
        self.assertTrue((files.dataframe[mask]['sample'] == 1).all())

    def test_031_get_mask_ignores_unknown_columns(self):
        files = self.made().initialize()
        self.assertEqual(int(files.get_mask({'nonexistent': 3}).sum()),
                         self.expected_files)

    def test_032_validate_finds_nothing_when_names_differ(self):
        files = self.made().initialize()
        self.assertEqual(files.validate_unique_metadata(verbose=False), {})

    def test_033_validate_finds_two_files_that_say_the_same(self):
        """Two names that carry the same metadata are an acquisition mistake."""
        write_data_file(self.root / "alpha" / "elsewhere" / "sample1_dose45_zone1_0.txt")

        files = self.made().initialize()
        duplicates = files.validate_unique_metadata(verbose=False)

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(next(iter(duplicates.values()))), 2)

    def test_034_finalize_applies_the_corrections(self):
        def keep_first_sample(df):
            return df[df['sample'] == 1]

        files = self.made().initialize()
        files.finalize([keep_first_sample])
        self.assertEqual(len(files.dataframe), 6)

    def test_035_finalize_refuses_a_method_that_returns_nothing(self):
        def forgets_to_return(df):
            df['new'] = 1

        files = self.made().initialize()
        with self.assertRaises(ValueError):
            files.finalize([forgets_to_return])

    # ---- reading the contents ---------------------------------------------

    def test_040_read_data_files(self):
        files = self.made().initialize()
        contents = files.read_data_files(reader_method=lambda path: Path(path).read_text())

        self.assertEqual(len(contents), self.expected_files)
        self.assertEqual(set(contents), set(files.dataframe.index))

    def test_041_read_only_a_part(self):
        files = self.made().initialize()
        mask = files.get_mask({'sample': 2})
        contents = files.read_data_files(reader_method=lambda path: None, mask=mask)

        self.assertEqual(len(contents), 6)

    # ---- the local copy ----------------------------------------------------

    def test_050_no_copy_to_start_with(self):
        self.assertFalse(self.made().has_valid_local_copy)

    def test_051_the_dataset_is_named_after_the_last_folder(self):
        self.assertEqual(self.made().dataset_name, 'data')
        self.assertEqual(self.made().local_root.name, 'data')

    def test_052_the_same_data_reached_two_ways_shares_one_copy(self):
        """
        Two people mount the same share differently. It is still one dataset,
        and must not be copied twice.
        """
        elsewhere = Path(self.temporary.name) / "another_mount"
        elsewhere.symlink_to(self.root.parent)

        one = DataFiles(self.root)
        other = DataFiles(elsewhere / "data")
        self.assertEqual(one.local_root, other.local_root)

    def test_053_a_marker_is_needed_not_just_a_folder(self):
        """A copy that was interrupted leaves a folder that looks perfectly fine."""
        files = self.made()
        files.local_root.mkdir(parents=True)
        write_data_file(files.local_root / "half_copied.txt")

        self.assertFalse(files.has_valid_local_copy)

    def test_054_marking_and_unmarking(self):
        files = self.made()
        files.mark_local_copy_as_valid()
        self.assertTrue(files.has_valid_local_copy)

        files.invalidate_local_copy()
        self.assertFalse(files.has_valid_local_copy)

    def test_055_invalidating_twice_is_harmless(self):
        files = self.made()
        files.invalidate_local_copy()
        files.invalidate_local_copy()

    def test_056_a_marker_from_another_dataset_is_refused(self):
        """
        Two datasets whose last folder is called the same thing share a folder,
        and reading the wrong one would return the measurements of another
        experiment without anything saying so.
        """
        files = self.made()
        files.local_root.mkdir(parents=True)
        (files.local_root / files.valid_marker).write_text("something_else\n/elsewhere\n")

        self.assertFalse(files.has_valid_local_copy)

    def test_057_the_marker_says_where_the_copy_came_from(self):
        files = self.made()
        files.mark_local_copy_as_valid()

        name, path = DataFiles.read_marker(files.local_root / files.valid_marker)
        self.assertEqual(name, 'data')
        self.assertEqual(path, files.root_signature)

    def test_058_copying_and_reading_the_copy(self):
        files = self.made()
        files.initialize()

        queue = deque()
        files.get_data_file_paths((queue,))
        files.copy_files_locally(queue)

        self.assertTrue(files.has_valid_local_copy)
        copied = list(files.local_root.rglob("*.txt"))
        self.assertEqual(len(copied), self.expected_files)

    def test_059_listing_and_deleting_the_copies(self):
        files = self.made()
        files.mark_local_copy_as_valid()

        copies = DataFiles.local_copies()
        self.assertEqual(len(copies), 1)
        folder, root, complete = copies[0]
        self.assertEqual(folder.name, 'data')
        self.assertEqual(root, files.root_signature)
        self.assertTrue(complete)

        files.delete_local_copy()
        self.assertEqual(DataFiles.local_copies(), [])
        self.assertFalse(files.local_root.exists())

    def test_05a_listing_when_nothing_was_ever_copied(self):
        self.assertEqual(DataFiles.local_copies(), [])


if __name__ == "__main__":
    unittest.main()
