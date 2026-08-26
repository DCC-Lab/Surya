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
import io
from contextlib import redirect_stdout
import warnings
from platformdirs import user_cache_path


def is_directory_usable(path, timeout=5):
    """
    Says whether a directory can really be read right now.

    Path.exists() is not enough for a network share. The system remembers what
    it was told about a folder, and keeps answering from memory long after the
    share stopped responding: exists() says yes, is_dir() says yes, and the
    first attempt to actually read something fails. Worse, a share that hangs
    rather than fails makes the reading block for as long as the mount decides,
    which can be minutes.

    So this asks for the first entry of the directory, which is the cheapest
    request that has to reach the other end, and it does so in a task of its
    own with a time limit. Whatever happens -- a missing folder, a share that
    answers with an error, a share that does not answer at all -- the answer
    comes back within `timeout` seconds and it is false.

    The task is left running if it never comes back: there is no way to
    interrupt a read stuck in the system, but being a background task it does
    not keep the program from ending.
    """
    answer = []

    def look():
        try:
            with os.scandir(path) as entries:
                next(iter(entries), None)  # empty is fine: it answered
            answer.append(True)
        except OSError:
            answer.append(False)

    thread = Thread(target=look, daemon=True)
    thread.start()
    thread.join(timeout)

    return bool(answer) and answer[0]


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

    def __init__(
        self, root=None, extensions=[".txt"], methods=None, metadata_patterns=None
    ):
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
        self.metadata_patterns = (
            metadata_patterns if metadata_patterns is not None else []
        )

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
                    print(
                        f"Warning: {self.local_root} holds a copy of '{name_found}', "
                        f"not of '{self.dataset_name}'. It will be ignored."
                    )
                    self.cache_warning_issued = True
                return False

            if path_found != self.root_signature and not self.cache_warning_issued:
                print(
                    f"Note: the local copy of '{self.dataset_name}' was made from "
                    f"'{path_found}', which is another way of reaching the same data."
                )
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
            root = self.root.absolute()  # the drive may be unreachable: do not block

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
            root = self.root.absolute()  # the drive may be unreachable: do not block

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

    def initialize(self, methods=None, create_local_copy=False):
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
        self.register_metadata_extraction_method(
            self.extract_extended_properties_from_path
        )

        threads = []
        queue = deque()
        copy_queue = deque()

        # if self.has_valid_local_copy:
        #     print(f"Delete {self.local_root / self.valid_marker} to avoid cache")

        threads.append(
            Thread(target=self.get_data_file_paths, args=((queue, copy_queue),))
        )
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
            copy_thread = Thread(target=self.copy_files_locally, args=(copy_queue,))
            copy_thread.start()  # Attempt to copy in the background

        self.dataframe = pd.DataFrame(self._properties)

        # The relative path of the file becomes the label of its row. The rows
        # are built by several tasks at once, so the order they end up in is
        # different every run: a plain row number would designate a different
        # file each time the program is started. The file name does not move.
        assert self.dataframe["file"].is_unique, "Two rows describe the same file"
        self.dataframe = self.dataframe.set_index("file")

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
                raise ValueError(
                    "The finalize method()s must return the final dataframe"
                )

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
                files += 1
                if (
                    dest_path.exists()
                    and dest_path.stat().st_size == absolute_path.stat().st_size
                ):
                    continue

                if not dest_path.parent.exists():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(
                    absolute_path, dest_path
                )  # copy2 preserves the modification dates

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
                    properties[key] = float(
                        re.sub(r"[\u00a0 ]", "", value).replace(",", ".")
                    )
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
        properties = {
            name: False
            for pattern in self.metadata_patterns
            for name in re.compile(pattern).groupindex
            if name.startswith("is_")
        }

        for pattern in self.metadata_patterns:
            match = re.search(pattern, file_path, re.IGNORECASE)
            if match is not None:
                properties.update(_to_normalized_values(match.groupdict()))

        # A time is spread over four groups that only mean something together.
        if properties.get("heure") is not None:
            properties["time"] = datetime.time(
                hour=properties["heure"],
                minute=properties["minutes"],
                second=properties["s"],
                microsecond=properties["ms"] * 1000,
            )

        properties["file"] = str(file_relative_path)

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

            extended_properties["size_in_bytes"] = file_info.st_size
            # Others possible
        except Exception as e:
            pass  # We just give up if unable to do it

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

                    properties = {"file": relative_path, "absolute_path": absolute_path}

                    for method in self.metadata_methods:
                        properties.update(method(root, relative_path))

                    with self._data_files_lock:
                        self._properties.append(properties)

                        if progress and time.time() > next_time:
                            print(f"Metadata from {len(self._properties)} files read")
                            next_time = time.time() + self.progress_delay

                else:
                    queue.appendleft(None)  # Put back for other tasks
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

            # Not exists(): a network share that stopped answering still looks
            # perfectly present, and the walk below would either return nothing
            # at all or hang for minutes. Better to say so here.
            if not is_directory_usable(root):
                raise ValueError(
                    f"The path {root} cannot be read. It may not exist, "
                    f"or it may be a network share that stopped answering."
                )

            next_progress_print = time.time() + 2
            for dirpath, dirs, files in os.walk(root):
                for name in files:
                    absolute_path = unicodedata.normalize(
                        "NFC", os.path.join(dirpath, name)
                    )
                    if Path(absolute_path).suffix not in self.extensions:
                        continue

                    # Hidden files, and anything inside a hidden folder. Testing
                    # every part of the path works on Windows too, where the
                    # separator is a backslash.
                    relative_parts = Path(absolute_path).relative_to(root).parts
                    if not invisible_files and any(
                        part.startswith(".") for part in relative_parts
                    ):
                        continue

                    # Files that belong to the machinery, not to the experiment:
                    # our own path cache, and the metadata companions that macOS
                    # scatters over network drives. They are not spectra, and
                    # they would show up as a row of missing values.
                    if name.startswith("._"):
                        continue

                    if progress and time.time() > next_progress_print:
                        print(".", end="", flush=True)
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

    def column_roles(self, mask=None, verbose=True):
        r"""
        Says what each column is worth, here, for telling one file from another.

        The same table describes several experiments, and a column that carries
        the whole meaning in one of them is often empty in the next: exp 1
        numbers its repetitions with 'indice2', which exp 2 and 3 never write at
        all. Before deciding what identifies a measurement, it is worth looking
        at what the names actually say in the part being worked on:

            files.column_roles(mask=files.get_mask({'exp': 1}))

        Two things are counted for every column, and they are facts rather than
        opinions:

          filled   : how many files say something about it
          distinct : how many different answers there are, an empty answer
                     counting as one of them

        Their ratio is what matters. A column with about as many answers as
        there are files -- the size of the file, the millisecond it was recorded
        -- can name a file but can never put two of them together: those belong
        in the `ignore` list of RamanData.averaged(). A column with a single
        answer says nothing here at all and can be left out of any fingerprint
        without changing a thing. What is left in between is what describes the
        measurements.

        Saying nothing comes about in two ways, which `filled` tells apart and
        which are worth different reactions. A column filled everywhere with the
        same answer is a question this part of the study did not vary: exp 2 is
        all raman, so 'modalite' says nothing there. A column filled nowhere is
        a question it never asked at all: exp 2 numbers nothing with 'indice2'.
        The first is a fact about the experiment, the second usually means the
        way files are named changed between experiments.

        The table comes back sorted with the most changeable column first, which
        is the order in which it reads best: the accidents at the top, what
        describes the experiment at the bottom.

        mask : an optional filter, as returned by get_mask(), to look at one
               part of the data. Without it, the whole table is described.
        """
        df = self.dataframe if mask is None else self.dataframe[mask]

        if len(df) == 0:
            raise ValueError("No file is selected: there is nothing to describe")

        roles = pd.DataFrame(
            {
                "filled": df.notna().sum(),
                "missing": df.isna().sum(),
                "distinct": df.nunique(dropna=False),
            }
        )
        roles["per_file"] = (roles["distinct"] / len(df)).round(3)
        roles["says_nothing"] = roles["distinct"] <= 1
        roles = roles.sort_values("per_file", ascending=False)
        roles.index.name = "column"

        if verbose:
            print(f"{len(df)} files, {len(roles)} columns")
            print(roles.to_string())

            silent = list(roles[roles["says_nothing"]].index)
            if silent:
                print(f"\n    {silent}")
                print(
                    f"    say nothing here -- one answer or none -- and can be left out of "
                    f"a fingerprint without changing anything"
                )

            # Not only the columns with exactly one answer per file: a column
            # with an answer for most of them is just as unusable, and those
            # are the ones that quietly wreck a fingerprint. The same rule is
            # used in validate_unique_metadata(), so the two reports agree.
            names_a_file = list(roles[roles["distinct"] > len(df) / 2].index)
            if names_a_file:
                print(f"\n    {names_a_file}")
                print(
                    f"    hold a different answer for nearly every file: they can name a "
                    f"file but can never put two of them together"
                )

        return roles

    def validate_unique_metadata(
        self,
        columns=None,
        mask=None,
        ignore=("absolute_path",),
        ignore_prefixes=("Spectrum:",),
        verbose=True,
        show=5,
    ):
        r"""
        Checks that a set of columns tells every file apart, and says what is
        missing when it does not.

        Two files carrying exactly the same metadata cannot be told apart
        afterwards. Called with nothing, this asks the question of the whole
        table and of all its columns, which is the acquisition check: a
        duplicate there usually means one measurement was named after another
        by mistake.

        Called with a set of columns, it asks a different and more useful
        question: do these columns identify a measurement? That is the question
        to answer before averaging anything, because whatever they fail to
        separate will be averaged together:

            files.validate_unique_metadata(
                columns=['petri', 'souris', 'zone', 'dose', 'indice1'],
                mask=files.get_mask({'exp': 2}))

        When some files are left indistinguishable, the report does not stop at
        saying so. It looks at those files alone and lists the other columns
        that DO differ between them, with the number of groups each one would
        separate. That list is the answer to "what is my identifier missing":

            680 of 1620 files are not told apart (340 groups)
                fixation     2 answers    330 groups
                is_adn       2 answers     10 groups

        Two kinds of failure are told apart, because they call for two very
        different things. A column that differs inside the groups is a column
        the identifier forgot: add it and the files separate. A column of the
        identifier that is empty for every one of those files is a name that
        never said it: nothing in the metadata can separate them, and it is the
        file names, or the choice of what to average, that has to change.

        columns         : the columns that are supposed to identify a file.
                          Without it, every column is used except those left
                          out below.
        mask            : an optional filter, as returned by get_mask(), to ask
                          the question of one part of the data. The three
                          experiments do not name themselves the same way and
                          are worth asking about separately.
        ignore          : column names to leave out. 'absolute_path' is left
                          out by default: it is not metadata, it is where the
                          file happens to sit on this machine, and since it
                          differs for every row it would hide every duplicate
                          there is -- and top every list of what differs
                          without ever saying anything.
        ignore_prefixes : whole families of columns to leave out. Instruments
                          add one column per setting they record, all sharing a
                          prefix, and those describe the measurement rather
                          than identifying the sample.
        show            : how many example groups to print.

        Returns a dictionary {signature: [list of files]} for the signatures
        that appear more than once, that is, the duplicates. An empty dictionary
        means the columns identify every file.
        """
        df = self.dataframe if mask is None else self.dataframe[mask]

        if len(df) == 0:
            raise ValueError("No file is selected: there is nothing to check")

        left_out = set(ignore)
        prefixes = tuple(ignore_prefixes)

        if columns is None:
            chosen = [
                c
                for c in df.columns
                if c not in left_out and not c.startswith(prefixes)
            ]
        else:
            unknown = [c for c in columns if c not in df.columns]
            if unknown:
                raise ValueError(
                    f"No such column: {unknown}. The columns available are: "
                    f"{list(df.columns)}"
                )
            chosen = list(columns)

        if not chosen:
            raise ValueError("No column is left to identify a file with")

        # dropna=False: a missing value is an answer like any other here. Two
        # files that both fail to say which mouse they come from are not thereby
        # different, they are exactly the pair this method exists to report.
        grouped = df.groupby(chosen, dropna=False, sort=False)

        duplicates = {}
        readable = {}
        for key, rows in grouped:
            if len(rows) < 2:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            # The missing answers are left out of the signature that is returned,
            # so that it holds what the names actually said. They are kept in the
            # line that gets printed, written as a dash, because a column that
            # said nothing is most of the reason two files look alike.
            signature = tuple(
                sorted(
                    ((c, v) for c, v in zip(chosen, key) if pd.notna(v)),
                    key=lambda pair: str(pair[0]),
                )
            )
            duplicates[signature] = list(rows.index)
            readable[signature] = " ".join(
                f"{c}={'-' if pd.isna(v) else v}" for c, v in zip(chosen, key)
            )

        if not verbose:
            return duplicates

        if not duplicates:
            print(f"{len(chosen)} columns tell all {len(df)} files apart")
            return duplicates

        ambiguous = df.loc[[name for names in duplicates.values() for name in names]]
        print(
            f"{len(ambiguous)} of {len(df)} files are not told apart by {chosen} "
            f"({len(duplicates)} groups)"
        )

        # What differs between files the chosen columns cannot separate is
        # exactly what the identifier is missing. Counting the groups each
        # column would separate ranks them by how much they would help.
        in_groups = ambiguous.groupby(chosen, dropna=False, sort=False)
        elsewhere = [
            c
            for c in df.columns
            if c not in chosen and c not in left_out and not c.startswith(prefixes)
        ]

        would_separate = {}
        for column in elsewhere:
            groups = int((in_groups[column].nunique(dropna=False) > 1).sum())
            if groups:
                would_separate[column] = (
                    groups,
                    int(ambiguous[column].nunique(dropna=False)),
                )

        if would_separate:
            print(
                f"\n    what differs between those files, and how many of the "
                f"{len(duplicates)} groups it would separate:"
            )
            for column, (groups, answers) in sorted(
                would_separate.items(), key=lambda item: -item[1][0]
            ):
                # A column with about as many answers as there are files is an
                # accident of the recording rather than a description of it: it
                # separates everything and means nothing.
                hint = (
                    "   <- nearly one answer per file"
                    if answers > len(ambiguous) / 2
                    else ""
                )
                print(
                    f"        {column:<20s} {answers:6d} answers   "
                    f"{groups:6d} groups{hint}"
                )

        never_said = {c: int(ambiguous[c].isna().sum()) for c in chosen}
        never_said = {c: n for c, n in never_said.items() if n}
        if never_said:
            print(f"\n    columns of the identifier that those files never said:")
            for column, how_many in sorted(
                never_said.items(), key=lambda item: -item[1]
            ):
                print(f"        {column:<20s} empty for {how_many} of {len(ambiguous)}")
            print(
                f"    where a column is empty there is nothing left to separate the files "
                f"with. That is not a missing column, it is a missing name: either the "
                f"files are renamed, or they are left out of the averaging."
            )

        print(f"\n    for instance:")
        for signature, names in list(duplicates.items())[:show]:
            print(f"        {len(names)} files share  {readable[signature]}")
            for name in names[:3]:
                print(f"            {name}")
            if len(names) > 3:
                print(f"            ... and {len(names) - 3} more")
        if len(duplicates) > show:
            print(f"        ... and {len(duplicates) - show} more groups")

        return duplicates

    def get_mask(self, mask_as_dict):
        df = self.dataframe
        mask = pd.Series(True, index=df.index)
        for key, value in mask_as_dict.items():
            if key not in df.columns:
                continue
            mask &= df[key].notna() & (df[key] == value)

        return mask

    def read_data_files(self, reader_method, mask=None):
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
            raise ValueError(
                "The dataframe index has duplicates: it cannot be used to match the files to their content"
            )

        files_data = {}
        next_time = time.time() + self.progress_delay
        for index, absolute_path in df["absolute_path"].items():
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
                    write_data_file(
                        folder / f"sample{sample}_dose45_zone{zone}_{number}.txt"
                    )
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

        self.assertEqual(files.dataframe.index.name, "file")
        self.assertTrue(files.dataframe.index.is_unique)
        self.assertIn(
            "sample1/zone1/sample1_dose45_zone1_0.txt",
            {str(Path(i).relative_to("alpha")) for i in files.dataframe.index},
        )

    def test_012_only_the_wanted_extensions(self):
        write_data_file(self.root / "alpha" / "notes.md")
        write_data_file(self.root / "alpha" / "table.csv")

        files = self.made().initialize()
        self.assertEqual(len(files.dataframe), self.expected_files)

    def test_013_several_extensions_at_once(self):
        write_data_file(self.root / "alpha" / "table.csv")

        files = self.made(extensions=[".txt", ".csv"]).initialize()
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

    def test_016_a_root_that_cannot_be_read(self):
        """
        A folder that is there but refuses to open is not a usable root.

        This is what a network share looks like once it stops answering: the
        system keeps saying the folder is there, because it remembers, and only
        an attempt to read finds out otherwise.
        """
        forbidden = Path(self.temporary.name) / "forbidden"
        forbidden.mkdir()
        write_data_file(forbidden / "inside.txt")
        os.chmod(forbidden, 0o000)
        try:
            self.assertTrue(forbidden.exists())  # it looks fine
            self.assertFalse(is_directory_usable(forbidden))

            with self.assertRaises(Exception):
                DataFiles(forbidden).initialize()
        finally:
            os.chmod(forbidden, 0o755)

    def test_017_a_directory_that_never_answers(self):
        """
        A share that hangs instead of failing must not hang the program.

        There is no way to interrupt a read stuck in the system, so the check
        gives up after a while and says no rather than waiting for a mount that
        may take minutes to decide.
        """

        class NeverAnswers:
            def __fspath__(self):
                time.sleep(3600)
                return "/"

        start = time.time()
        self.assertFalse(is_directory_usable(NeverAnswers(), timeout=0.5))
        self.assertLess(time.time() - start, 5)

    def test_018_an_empty_directory_is_usable(self):
        """Having nothing in it is not the same as being unreachable."""
        empty = Path(self.temporary.name) / "empty"
        empty.mkdir()
        self.assertTrue(is_directory_usable(empty))

    # ---- reading the names -----------------------------------------------

    def test_020_patterns_fill_the_columns(self):
        files = self.made().initialize()
        row = files.dataframe.iloc[0]

        self.assertIn(row["sample"], (1, 2))
        self.assertEqual(row["dose"], 45)
        self.assertEqual(row["mode"], "alpha")

    def test_021_a_whole_number_stays_a_whole_number(self):
        """45 must not become 45.0: it exports badly and reads worse."""
        files = self.made().initialize()
        self.assertIsInstance(files.dataframe.iloc[0]["sample"], (int, np.integer))

    def test_022_leading_zeros_are_still_numbers(self):
        write_data_file(self.root / "alpha" / "sample007_dose45_zone1_0.txt")

        files = self.made().initialize()
        samples = set(files.dataframe["sample"].dropna())
        self.assertIn(7, samples)

    def test_023_a_decimal_written_the_french_way(self):
        """Computers set to French write 2,5 where others write 2.5."""
        write_data_file(self.root / "alpha" / "sample3_dose2,5_zone1_0.txt")

        files = self.made().initialize()
        doses = set(files.dataframe["dose"].dropna())
        self.assertIn(2.5, doses)

    def test_024_words_are_lowercased(self):
        """The same word is spelled two ways across folders, never three."""
        write_data_file(self.root / "BETA" / "sample9_dose45_zone1_0.txt")

        files = self.made().initialize()
        self.assertEqual(set(files.dataframe["mode"].dropna()), {"alpha", "beta"})

    def test_025_presence_groups_are_true_or_false(self):
        write_data_file(self.root / "alpha" / "test" / "sample8_dose45_zone1_0.txt")

        files = self.made().initialize()
        df = files.dataframe

        self.assertEqual(df["is_test"].dtype, bool)
        self.assertEqual(int(df["is_test"].sum()), 1)

    def test_026_a_presence_group_is_never_missing(self):
        """
        A column that is sometimes false and sometimes missing cannot be
        filtered on: a hole is neither true nor false, so a row holding one is
        dropped by a test for false just as surely as a row holding true.
        """
        files = self.made().initialize()
        df = files.dataframe

        self.assertEqual(int(df["is_reference"].isna().sum()), 0)
        self.assertEqual(int(df["is_reference"].sum()), 0)

    def test_027_extraction_methods_are_merged_in(self):
        def extra(root, relative_path):
            return {"extra": "yes"}

        files = self.made(methods=[extra]).initialize()
        self.assertTrue((files.dataframe["extra"] == "yes").all())

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
        mask = files.get_mask({"sample": 1})

        self.assertEqual(int(mask.sum()), 6)
        self.assertTrue((files.dataframe[mask]["sample"] == 1).all())

    def test_031_get_mask_ignores_unknown_columns(self):
        files = self.made().initialize()
        self.assertEqual(
            int(files.get_mask({"nonexistent": 3}).sum()), self.expected_files
        )

    def test_032_validate_finds_nothing_when_names_differ(self):
        files = self.made().initialize()
        self.assertEqual(files.validate_unique_metadata(verbose=False), {})

    def test_033_validate_finds_two_files_that_say_the_same(self):
        """Two names that carry the same metadata are an acquisition mistake."""
        write_data_file(
            self.root / "alpha" / "elsewhere" / "sample1_dose45_zone1_0.txt"
        )

        files = self.made().initialize()
        duplicates = files.validate_unique_metadata(verbose=False)

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(next(iter(duplicates.values()))), 2)

    def test_034_finalize_applies_the_corrections(self):
        def keep_first_sample(df):
            return df[df["sample"] == 1]

        files = self.made().initialize()
        files.finalize([keep_first_sample])
        self.assertEqual(len(files.dataframe), 6)

    def test_035_finalize_refuses_a_method_that_returns_nothing(self):
        def forgets_to_return(df):
            df["new"] = 1

        files = self.made().initialize()
        with self.assertRaises(ValueError):
            files.finalize([forgets_to_return])

    # ---- what each column is worth ------------------------------------------

    def report_of(self, method, *args, **kwargs):
        """Runs a method for what it prints rather than for what it returns."""
        printed = io.StringIO()
        with redirect_stdout(printed):
            method(*args, **kwargs)
        return printed.getvalue()

    def test_060_one_row_per_column(self):
        files = self.made().initialize()
        roles = files.column_roles(verbose=False)

        self.assertEqual(sorted(roles.index), sorted(files.dataframe.columns))
        self.assertEqual(int(roles.loc["sample", "filled"]), self.expected_files)
        self.assertEqual(int(roles.loc["sample", "distinct"]), 2)

    def test_061_a_column_with_one_answer_says_nothing(self):
        """Every file here was written in mode alpha, so the column is useless."""
        files = self.made().initialize()
        roles = files.column_roles(verbose=False)

        self.assertTrue(roles.loc["mode", "says_nothing"])
        self.assertFalse(roles.loc["zone", "says_nothing"])

    def test_062_a_column_that_names_a_file_can_group_nothing(self):
        files = self.made().initialize()
        roles = files.column_roles(verbose=False)

        self.assertEqual(
            int(roles.loc["absolute_path", "distinct"]), self.expected_files
        )
        self.assertEqual(roles.loc["absolute_path", "per_file"], 1.0)

    def test_062a_a_column_that_names_nearly_every_file_is_named_too(self):
        """
        Not only the columns with exactly one answer per file. A column with an
        answer for most of them is just as unable to group anything, and those
        are the ones that quietly ruin a fingerprint.
        """
        for number in range(self.expected_files):
            write_data_file(
                self.root / "alpha" / f"sample3_dose45_zone1_{number}.txt",
                lines=tuple(f"{i}.0 {i}.0" for i in range(number + 1)),
            )

        files = self.made().initialize()
        report = self.report_of(files.column_roles)

        self.assertIn("nearly every file", report)
        hint = report[
            report.index("nearly every file") - 400 : report.index("nearly every file")
        ]
        self.assertIn("size_in_bytes", hint)

    def test_063_the_most_changeable_column_comes_first(self):
        files = self.made().initialize()
        roles = files.column_roles(verbose=False)

        self.assertEqual(roles.index[0], "absolute_path")
        self.assertTrue(roles["per_file"].is_monotonic_decreasing)

    def test_064_a_column_can_say_nothing_in_one_part_only(self):
        """
        The whole point of the mask: what a column is worth depends on where you
        look. 'sample' tells the files apart, until you look at one sample.
        """
        files = self.made().initialize()

        self.assertFalse(
            files.column_roles(verbose=False).loc["sample", "says_nothing"]
        )
        one = files.column_roles(mask=files.get_mask({"sample": 1}), verbose=False)
        self.assertTrue(one.loc["sample", "says_nothing"])
        self.assertEqual(int(one["filled"].max()), self.expected_files // 2)

    def test_065_describing_nothing_at_all(self):
        files = self.made().initialize()
        with self.assertRaises(ValueError):
            files.column_roles(mask=files.get_mask({"sample": 99}), verbose=False)

    # ---- is this what identifies a file? ------------------------------------

    def test_066_a_chosen_set_of_columns_that_is_not_enough(self):
        """The three repetitions of a zone are alike until 'number' is added."""
        files = self.made().initialize()

        duplicates = files.validate_unique_metadata(
            columns=["sample", "zone"], verbose=False
        )
        self.assertEqual(len(duplicates), 4)  # 2 samples x 2 zones
        self.assertTrue(all(len(names) == 3 for names in duplicates.values()))

        enough = files.validate_unique_metadata(
            columns=["sample", "zone", "number"], verbose=False
        )
        self.assertEqual(enough, {})

    def test_067_the_report_names_the_missing_column(self):
        """
        What the tool is for: not 'these files look alike', but 'here is the
        column that would tell them apart'.
        """
        files = self.made().initialize()
        report = self.report_of(
            files.validate_unique_metadata, columns=["sample", "zone"]
        )

        self.assertIn("what differs", report)
        self.assertIn("number", report)
        self.assertNotIn("absolute_path", report)

    def test_068_the_report_names_a_column_the_names_never_said(self):
        """
        The other kind of failure: nothing is missing from the identifier, the
        files themselves never said which zone they came from. No column can be
        added to fix that.
        """
        write_data_file(self.root / "alpha" / "sample1_dose45_0.txt")
        write_data_file(self.root / "alpha" / "aside" / "sample1_dose45_0.txt")

        files = self.made().initialize()
        report = self.report_of(
            files.validate_unique_metadata, columns=["sample", "zone", "number"]
        )

        self.assertIn("never said", report)
        self.assertIn("zone", report)

    def test_069_the_question_can_be_asked_of_one_part_only(self):
        files = self.made().initialize()
        duplicates = files.validate_unique_metadata(
            columns=["sample", "zone"],
            mask=files.get_mask({"sample": 1}),
            verbose=False,
        )
        self.assertEqual(len(duplicates), 2)  # the two zones of sample 1

    def test_06a_a_misspelled_column_is_refused(self):
        files = self.made().initialize()
        with self.assertRaises(ValueError):
            files.validate_unique_metadata(columns=["sample", "zonne"], verbose=False)

    def test_06b_asking_about_nothing_at_all(self):
        files = self.made().initialize()
        with self.assertRaises(ValueError):
            files.validate_unique_metadata(
                mask=files.get_mask({"sample": 99}), verbose=False
            )
        with self.assertRaises(ValueError):
            files.validate_unique_metadata(columns=[], verbose=False)

    def test_06c_the_signature_holds_what_the_names_said(self):
        """The values that were missing are left out of the signature."""
        files = self.made().initialize()
        duplicates = files.validate_unique_metadata(
            columns=["sample", "zone"], verbose=False
        )

        signature = next(iter(duplicates))
        self.assertEqual(sorted(key for key, _ in signature), ["sample", "zone"])

    # ---- reading the contents ---------------------------------------------

    def test_040_read_data_files(self):
        files = self.made().initialize()
        contents = files.read_data_files(
            reader_method=lambda path: Path(path).read_text()
        )

        self.assertEqual(len(contents), self.expected_files)
        self.assertEqual(set(contents), set(files.dataframe.index))

    def test_041_read_only_a_part(self):
        files = self.made().initialize()
        mask = files.get_mask({"sample": 2})
        contents = files.read_data_files(reader_method=lambda path: None, mask=mask)

        self.assertEqual(len(contents), 6)

    # ---- the local copy ----------------------------------------------------

    def test_050_no_copy_to_start_with(self):
        self.assertFalse(self.made().has_valid_local_copy)

    def test_051_the_dataset_is_named_after_the_last_folder(self):
        self.assertEqual(self.made().dataset_name, "data")
        self.assertEqual(self.made().local_root.name, "data")

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
        (files.local_root / files.valid_marker).write_text(
            "something_else\n/elsewhere\n"
        )

        self.assertFalse(files.has_valid_local_copy)

    def test_057_the_marker_says_where_the_copy_came_from(self):
        files = self.made()
        files.mark_local_copy_as_valid()

        name, path = DataFiles.read_marker(files.local_root / files.valid_marker)
        self.assertEqual(name, "data")
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
        self.assertEqual(folder.name, "data")
        self.assertEqual(root, files.root_signature)
        self.assertTrue(complete)

        files.delete_local_copy()
        self.assertEqual(DataFiles.local_copies(), [])
        self.assertFalse(files.local_root.exists())

    def test_05a_listing_when_nothing_was_ever_copied(self):
        self.assertEqual(DataFiles.local_copies(), [])


if __name__ == "__main__":
    unittest.main()
