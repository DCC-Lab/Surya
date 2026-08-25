import re
from pathlib import Path

import numpy as np
import pandas as pd
from datafiles import DataFiles
import unittest
import tempfile


class RamanData:
    r"""
    Raman spectra recorded with the Ocean Optics QEPro, ready for analysis.

    This class is the half of the work that knows what a spectrum is. The other
    half, DataFiles, knows how to find files on a slow network drive, keep a
    local copy of them, read them with several tasks at once and describe them
    in a table. DataFiles is deliberately kept general: tomorrow the files could
    be images instead of spectra, and it would not have to change.

    So the two are used together rather than merged: RamanData supplies the
    knowledge of the QEPro file format, DataFiles supplies the reading itself.

        files = DataFiles(root, metadata_patterns=[r"sample(?P<sample>\d+)",
                                                   r"dose(?P<dose>\d+)"])
        files.initialize()

        raman = RamanData(files).initialize(mask=files.get_mask({'dose': 0}))
        X, y = raman.training_set('dose')

    What initialize() produces is exactly the shape that PCA, LDA and PLS expect --
    those live in scikit-learn, not in scipy:

        X    : one row per spectrum, one column per measured point, float64
        meta : the rows of the metadata table describing those spectra, in the
               very same order as the rows of X
        axis : the wavelengths, kept once instead of once per spectrum

    The order is what matters most. X and meta are both derived from a single
    list of file names, so row i of X is always described by row i of meta.
    Building the two separately is the classic way of training a model on
    shuffled labels, and nothing warns you when it happens.
    """

    # One data line of a QEPro file: a wavelength, a separator, an intensity,
    # and nothing else. The whole file is matched in one go rather than line by
    # line, which is what makes reading fast enough for thousands of files.
    SPECTRUM_LINE = re.compile(rb"^[ \t]*(\d+[.,]?\d*)[ \t,;]+(-?\d*[.,]?\d*)[ \t]*\r?$",
                               re.MULTILINE)

    def __init__(self, files=None):
        """
        files : the DataFiles that found and described the spectra. It may be
                left out when the data is going to be read back from disk with
                RamanData.read().
        """
        self.files = files
        self.X = None
        self.meta = None
        self.axis = None

        # Filled in by initialize(), and read back by report()
        self.files_offered = 0
        self.expected_length = None
        self.rejected = {}

    def __len__(self):
        return 0 if self.X is None else self.X.shape[0]

    @property
    def shape(self):
        """(number of spectra, number of points), or None before initialize()."""
        return None if self.X is None else self.X.shape

    @staticmethod
    def extract_header_from_file(root, relative_path):
        """
        Since we use Ocean Optics Raman QEPro, we read the header of the 
        file_path a extract the metadata from the header, which we return in a
        dictionary

        """

        file_path = Path(root) / Path(relative_path)

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

    @staticmethod
    def read_spectrum(absolute_path):
        """
        Reads one QEPro file straight into two numpy arrays.

        This is the only place that knows what the inside of a spectrum file
        looks like. It is handed to DataFiles.read_data_files(), which takes
        care of walking the list, showing progress and using the local copy.

        Returns (wavelength, intensity), both float64, both empty if the file
        holds no data line at all -- an empty measurement, or a laboratory note
        that happens to have been saved with a .txt name.
        """
        with open(absolute_path, "rb") as data_file:
            raw = data_file.read()

        # Some computers with French settings write a comma as the decimal mark.
        values = RamanData.SPECTRUM_LINE.findall(raw.replace(b",", b"."))
        if not values:
            return np.empty(0), np.empty(0)

        columns = np.array(values, dtype=np.float64)
        return columns[:, 0], columns[:, 1]

    def initialize(self, mask=None, verbose=True):
        """
        Reads every spectrum and stacks them into one matrix.

        Spectra that do not have the same number of points as the majority are
        left out and reported: an empty file, or a measurement made with another
        instrument, cannot be a row of the same matrix. Spectra that have the
        right number of points but a different wavelength axis stop the whole
        thing, because stacking them would quietly compare one wavelength to
        another and every result afterwards would be meaningless.

        mask : an optional filter, as returned by DataFiles.get_mask(), to read
               only some of the spectra.

        Returns self, so that the call can be chained.
        """
        if self.files is None:
            raise ValueError("No DataFiles was given: nothing to read")

        spectra = self.files.read_data_files(reader_method=self.read_spectrum, mask=mask)

        lengths = {}
        for index, (wavelength, intensity) in spectra.items():
            lengths.setdefault(len(intensity), []).append(index)

        if not lengths:
            raise ValueError("No spectrum was read")

        # The majority length wins: it is the instrument's normal output.
        expected_length = max(lengths, key=lambda n: len(lengths[n]))
        usable = lengths[expected_length]
        rejected = [i for n, group in lengths.items() if n != expected_length for i in group]

        # What went in and what came out, kept so that it can be looked at long
        # after initialize() has printed and scrolled away.
        self.files_offered = len(spectra)
        self.expected_length = expected_length
        self.rejected = {index: len(spectra[index][1]) for index in rejected}

        axis = spectra[usable[0]][0]
        for index in usable:
            if not np.array_equal(spectra[index][0], axis):
                raise ValueError(f"{index} has the same number of points but not the same "
                                 f"wavelengths: the spectra cannot be stacked without being "
                                 f"interpolated onto a common axis first")

        self.X = np.vstack([spectra[index][1] for index in usable])
        self.meta = self.files.dataframe.loc[usable]
        self.axis = axis

        assert self.X.shape[0] == len(self.meta), "X and meta must describe the same spectra"

        if verbose:
            self.report()

        return self

    def report(self, verbose=True, show=10):
        """
        Says how many files were on offer and how many spectra came out of them.

        The two numbers are rarely the same, and the difference is worth
        looking at rather than discovering months later: a file that holds no
        measurement, or one recorded with another instrument, cannot become a
        row of the matrix and is quietly left aside. This is where those files
        are named.

        Returns the counts as a dictionary, so that a script can check them:

            offered  : how many files DataFiles handed over
            kept     : how many of them became a row of X
            rejected : how many were left out
            points   : the number of points a spectrum is expected to have
            lengths  : how many rejected files had each length, the length 0
                       meaning a file that holds no measurement at all

        show : how many rejected file names to print. None prints them all.
        """
        kept = 0 if self.X is None else self.X.shape[0]

        lengths = {}
        for length in self.rejected.values():
            lengths[length] = lengths.get(length, 0) + 1

        counts = {"offered": self.files_offered, "kept": kept,
                  "rejected": len(self.rejected), "points": self.expected_length,
                  "lengths": lengths}

        if not verbose:
            return counts

        print(f"{counts['offered']} files offered, {counts['kept']} spectra of "
              f"{counts['points']} points kept, {counts['rejected']} left out")

        for length, how_many in sorted(lengths.items()):
            reason = "hold no measurement" if length == 0 else f"have {length} points"
            print(f"    {how_many:5d} files {reason}")

        names = list(self.rejected)
        for index in names if show is None else names[:show]:
            print(f"        {self.rejected[index]:5d} points  {index}")
        if show is not None and len(names) > show:
            print(f"        ... and {len(names) - show} more")

        return counts

    def training_set(self, column, dropna=True):
        """
        Returns (X, y) ready to be handed to a scikit-learn estimator.

        Two traps are taken care of here. First, a column of whole numbers that
        accepts missing values ('Int64') turns into an array of Python objects,
        which scikit-learn refuses; the values are converted to decimals.
        Second, the rows without a label have to disappear from X and from y at
        the same time, which is easy to get wrong when the two are filtered in
        separate statements.

        column : the metadata column to predict, for instance 'dose'.
        """
        if self.X is None:
            raise ValueError("Nothing has been read yet: call initialize() first")

        labels = self.meta[column]
        if not dropna:
            return self.X, labels.astype("float64").to_numpy()

        known = labels.notna().to_numpy()
        return self.X[known], labels[known].astype("float64").to_numpy()

    def groups(self, column):
        """
        Returns the group of every spectrum, for a grouped cross-validation.

        Ten spectra of the same subject are not ten independent measurements. If
        they are split at random between the training set and the test set, the
        model recognizes the subject rather than the effect being studied, and
        the score comes out far too good. Passing these groups to GroupKFold
        keeps all the spectra of one subject on the same side of the split.

        column : the metadata column that says which subject a spectrum belongs
                 to -- one mouse, one patient, one sample, depending on the
                 study.
        """
        if self.meta is None:
            raise ValueError("Nothing has been read yet: call initialize() first")

        return self.meta[column].to_numpy()

    def averaged(self, ignore=(), on=None, verbose=True):
        r"""
        Averages together the spectra that describe the same measurement.

        Ten acquisitions of one spot on one sample are not ten independent
        measurements: they differ only by their repetition number, and averaging
        them is what removes the noise without pretending to have taken more
        data than was actually taken.

        What counts as "the same measurement" is decided by a fingerprint: the
        values of a set of metadata columns. Two spectra whose fingerprints are
        equal are averaged together. The fingerprint is normally described by
        what it leaves OUT, that is, by the columns that change from one
        repetition to the next:

            mean = raman.averaged(ignore=['indice1', 'heure', 'minutes', 's',
                                          'ms', 'size_in_bytes'])

        Everything else -- souris, petri, jour, dose, zone, ... -- stays in the
        fingerprint, so nothing is ever merged that differs in a way which was
        not explicitly declared uninteresting. The other way round is there for
        when it is shorter to say what matters than what does not:

            mean = raman.averaged(on=['souris', 'zone', 'dose'])

        Nothing is decided for you about which columns to ignore, because this
        class cannot know what anyone is measuring. A column that happens to
        hold a different value for every file -- the size of the file, the
        second at which it was recorded -- leaves every spectrum alone in a
        group of its own, and the method then does nothing at all. The printed
        summary says how many rows came out alone and names the columns
        responsible, so that they can be added to `ignore`.

        The result is a new RamanData; this one is left untouched.

          X    : one row per group, the average of the spectra of that group
          axis : the same wavelengths, which averaging does not change
          meta : one row per group. The fingerprint columns hold the values the
                 group shares. A column left out of the fingerprint keeps its
                 value when the whole group agrees on it, and is left empty when
                 the group does not: there is no single acquisition time for ten
                 acquisitions. One column is added, `n_averaged`, saying how
                 many spectra went into the row.

        The rows are labelled by their fingerprint rather than by a file name,
        since a row no longer comes from one file.

        Averaging an already averaged dataset averages averages, which stops
        being the same as averaging everything at once as soon as the groups do
        not all hold the same number of spectra. `n_averaged` is written down
        but deliberately not used as a weight: do the grouping you want in a
        single call.

        ignore  : the columns to leave out of the fingerprint.
        on      : the columns that make up the fingerprint, given directly.
                  Cannot be given together with `ignore`.
        verbose : whether to print how many spectra became how many rows.
        """
        if self.X is None:
            raise ValueError("Nothing has been read yet: call initialize() first")

        if on is not None and len(ignore) > 0:
            raise ValueError("Give either `on` or `ignore`, not both: each is the other's "
                             "opposite, so giving both can only say the same thing twice "
                             "or contradict it")

        # The path of a file is the file itself, not a property of what was
        # measured. It differs for every spectrum by construction, so keeping it
        # in the fingerprint could only ever leave every group alone.
        columns = [column for column in self.meta.columns if column != 'absolute_path']

        if on is not None:
            asked, fingerprint = list(on), list(on)
        else:
            asked = list(ignore)
            fingerprint = [column for column in columns if column not in set(ignore)]

        unknown = [column for column in asked if column not in columns]
        if unknown:
            raise ValueError(f"No such column: {unknown}. A misspelled name would quietly "
                             f"change the fingerprint instead of failing, so it is refused "
                             f"here. The columns available are: {columns}")

        meta = self.meta[columns]

        # An empty fingerprint says that every spectrum describes the same
        # measurement. A constant column says so without needing a special case.
        everything = "_everything"
        if not fingerprint:
            meta = meta.assign(**{everything: 0})
            grouping = [everything]
        else:
            grouping = fingerprint

        # dropna=False is not a detail. With the default, a single missing value
        # anywhere in the fingerprint makes the spectrum disappear from the
        # result without a word, and half the columns are empty for a
        # calibration file. sort=False keeps the groups in the order they first
        # appear, which is what makes ngroup() below agree, row for row, with
        # the table built from first().
        grouped = meta.groupby(grouping, dropna=False, sort=False)

        codes = grouped.ngroup().to_numpy()
        n_groups = int(codes.max()) + 1

        # first() gives the value the group shares; where the group does not
        # agree, mask() empties the cell rather than keeping the value of one
        # row and presenting it as if it described all of them.
        summary = grouped.first().mask(grouped.nunique(dropna=False) > 1).reset_index()
        if everything in summary.columns:
            summary = summary.drop(columns=[everything])

        assert len(summary) == n_groups, "there must be exactly one row per group"

        X = np.vstack([self.X[codes == group].mean(axis=0) for group in range(n_groups)])
        summary['n_averaged'] = np.bincount(codes, minlength=n_groups)

        labels = [" ".join(f"{column}={row[column]}" for column in fingerprint) or "all"
                  for _, row in summary.iterrows()]
        summary.index = pd.Index(labels, name='fingerprint')

        if not summary.index.is_unique:
            raise ValueError("Two different groups end up with the same fingerprint once it is "
                             "written out. That happens when a column holds values that look "
                             "alike as text; leave that column out of the fingerprint.")

        result = RamanData()
        result.X = X
        result.axis = self.axis
        result.meta = summary
        result.expected_length = self.expected_length
        result.files_offered = len(self)

        assert result.X.shape[0] == len(result.meta), "X and meta must describe the same rows"

        if verbose:
            alone = int((summary['n_averaged'] == 1).sum())
            largest = int(summary['n_averaged'].max())
            print(f"{len(self)} spectra averaged into {n_groups} rows "
                  f"({alone} alone, largest group {largest})")

            # A fingerprint that leaves most rows alone is nearly always a
            # forgotten column rather than a real result. Rather than leaving
            # that to be hunted down by hand, each column is taken out of the
            # fingerprint in turn to see whether that alone would merge rows.
            # The ones that would are named: they are what is keeping the
            # spectra apart. This costs one grouping per column, which is why
            # it is only done when the result looks wrong.
            if alone > n_groups / 2:
                culprits = []
                for column in fingerprint:
                    rest = [other for other in grouping if other != column]
                    if not rest:
                        continue
                    without = int(meta.groupby(rest, dropna=False, sort=False).ngroup().max()) + 1
                    if without < n_groups:
                        culprits.append(column)

                if culprits:
                    print(f"    most rows came out alone. Leaving out any of {culprits} "
                          f"would merge some of them: those columns change between "
                          f"repetitions and most likely belong in `ignore`")

        return result

    def save(self, path):
        """
        Writes the matrix and its metadata so that they cannot drift apart.

        The file names are saved next to X, so that reading back can check that
        the two files still describe the same spectra in the same order. Without
        that check, regenerating one half of the analysis and not the other
        gives a silently mismatched dataset.
        """
        if self.X is None:
            raise ValueError("Nothing has been read yet: call initialize() first")

        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), X=self.X, axis=self.axis,
                            files=np.array(list(self.meta.index), dtype=object),
                            # kept so that report() still works months later,
                            # when what was left out matters more than ever
                            rejected_files=np.array(list(self.rejected), dtype=object),
                            rejected_points=np.array(list(self.rejected.values()), dtype=int),
                            files_offered=self.files_offered)
        self.meta.to_pickle(path.with_suffix(".pkl"))
        return path

    @classmethod
    def read(cls, path):
        """Reads back what save() wrote, and checks the alignment."""
        path = Path(path)
        stored = np.load(path.with_suffix(".npz"), allow_pickle=True)
        meta = pd.read_pickle(path.with_suffix(".pkl"))

        if list(stored["files"]) != list(meta.index):
            raise ValueError("The matrix and the metadata no longer describe the same spectra")

        raman = cls()
        raman.X = stored["X"]
        raman.axis = stored["axis"]
        raman.meta = meta

        # Files written before these counts existed simply have none.
        if "rejected_files" in stored:
            raman.rejected = dict(zip(stored["rejected_files"],
                                      (int(n) for n in stored["rejected_points"])))
            raman.files_offered = int(stored["files_offered"])
        raman.expected_length = raman.X.shape[1]

        return raman


def write_spectrum_file(path, points=8, first=100.0, step=10.0, decimal=".",
                        header=True, intensity=lambda i: 1000.0 + i):
    """
    Writes a file that looks like what the QEPro produces.

    The tests need spectra whose every value is known in advance, which the real
    data cannot give. Everything that varies from one machine to another is a
    parameter here: the decimal mark, the presence of a header, the number of
    points.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if header:
        lines += [f"Data from {path.name} Node", "",
                  "Date: Tue Jun 30 14:00:00 CEST 2026",
                  "Integration Time (sec): 1",
                  ">>>>>Begin Spectral Data<<<<<"]

    for i in range(points):
        wavelength = f"{first + i * step:.3f}".replace(".", decimal)
        value = f"{intensity(i):.3f}".replace(".", decimal)
        lines.append(f"{wavelength}\t{value}")

    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


class TestRamanData(unittest.TestCase):
    """
    Tests for RamanData, run against spectra written by the tests themselves.

    Nothing here touches the network drive or the real measurements. A handful
    of small files are written into a temporary folder, which makes the tests
    fast, repeatable, and able to describe on purpose the awkward cases that
    only turn up once in the real data: an empty file, a spectrum recorded with
    another instrument, a comma used as the decimal mark.

    The local copy of DataFiles is redirected into that same temporary folder.
    Without this, a valid copy left over from a real run would be used instead
    of the files written here, and the tests would silently check the wrong
    data.
    """

    POINTS = 8

    # The metadata of the made-up files below. Deliberately not the metadata of
    # any real study: RamanData must work for whatever the file names happen to
    # say, so the tests describe a sample, a dose and a zone and nothing more.
    PATTERNS = [
        r"sample(?P<sample>\d+)",
        r"dose(?P<dose>\d+)",
        r"zone(?P<zone>\d+)",
    ]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "data"

        # Two samples, two zones each, five spectra per zone: enough to
        # exercise masks and groups without making the tests slow.
        self.expected_spectra = 0
        for sample in (1, 2):
            dose = 0 if sample == 1 else 45
            for zone in (1, 2):
                folder = self.root / f"sample{sample}" / f"zone{zone}"
                for number in range(5):
                    name = f"sample{sample}_dose{dose}_zone{zone}_{number}.txt"
                    write_spectrum_file(folder / name, points=self.POINTS,
                                        intensity=lambda i, s=sample: 1000.0 + 100 * s + i)
                    self.expected_spectra += 1

        # DataFiles keeps its local copies under one folder shared by every
        # instance. It is moved aside for the duration of the test, so that the
        # tests can never write into, or read from, a real copy.
        self.saved_cache_root = DataFiles.cache_root
        DataFiles.cache_root = Path(self.temporary.name) / "cache"

        self.datafiles = DataFiles(self.root, metadata_patterns=self.PATTERNS)
        self.assertIsNotNone(self.datafiles)

    def tearDown(self):
        DataFiles.cache_root = self.saved_cache_root
        self.temporary.cleanup()

    def initialized(self, mask=None):
        """A RamanData that has read the files written by setUp()."""
        self.datafiles.initialize()
        return RamanData(self.datafiles).initialize(mask=mask, verbose=False)

    # ---- creating -------------------------------------------------------

    def test_001_init(self):
        self.assertIsNotNone(RamanData(self.datafiles))

    def test_002_init_without_datafiles(self):
        """An empty RamanData is legal: RamanData.read() fills it from disk."""
        self.assertIsNotNone(RamanData())

    def test_003_nothing_is_read_yet(self):
        raman = RamanData(self.datafiles)
        self.assertIsNone(raman.X)
        self.assertIsNone(raman.shape)
        self.assertEqual(len(raman), 0)

    # ---- reading one file ----------------------------------------------

    def test_010_read_one_spectrum(self):
        path = write_spectrum_file(self.root / "one.txt", points=4, first=100.0, step=10.0)
        wavelength, intensity = RamanData.read_spectrum(path)

        self.assertEqual(len(wavelength), 4)
        self.assertEqual(len(intensity), 4)
        self.assertEqual(wavelength.dtype, np.float64)
        np.testing.assert_allclose(wavelength, [100.0, 110.0, 120.0, 130.0])
        np.testing.assert_allclose(intensity, [1000.0, 1001.0, 1002.0, 1003.0])

    def test_011_header_is_not_taken_for_data(self):
        """The lines before the data must not become points of the spectrum."""
        with_header = write_spectrum_file(self.root / "with.txt", points=4, header=True)
        without = write_spectrum_file(self.root / "without.txt", points=4, header=False)

        np.testing.assert_array_equal(RamanData.read_spectrum(with_header)[1],
                                      RamanData.read_spectrum(without)[1])

    def test_012_comma_as_decimal_mark(self):
        """Computers set to French write 100,000 where others write 100.000."""
        french = write_spectrum_file(self.root / "fr.txt", points=4, decimal=",")
        english = write_spectrum_file(self.root / "en.txt", points=4, decimal=".")

        np.testing.assert_allclose(RamanData.read_spectrum(french)[0],
                                   RamanData.read_spectrum(english)[0])

    def test_013_file_without_any_data(self):
        """A laboratory note saved as .txt gives an empty spectrum, not an error."""
        note = self.root / "note left by the operator, not a measurement.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("rien a voir avec un spectre\n", encoding="latin-1")

        wavelength, intensity = RamanData.read_spectrum(note)
        self.assertEqual(len(wavelength), 0)
        self.assertEqual(len(intensity), 0)

    def test_014_negative_intensities_are_kept(self):
        """After a baseline subtraction, an intensity can be below zero."""
        path = write_spectrum_file(self.root / "negative.txt", points=4,
                                   intensity=lambda i: -5.0 - i)
        _, intensity = RamanData.read_spectrum(path)
        np.testing.assert_allclose(intensity, [-5.0, -6.0, -7.0, -8.0])

    def test_015_windows_line_endings(self):
        """Files written on Windows end their lines with two characters."""
        path = self.root / "crlf.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Data from x Node\r\n>>>>>Begin Spectral Data<<<<<\r\n"
                         b"100.000\t1000.000\r\n110.000\t1001.000\r\n")

        wavelength, intensity = RamanData.read_spectrum(path)
        np.testing.assert_allclose(wavelength, [100.0, 110.0])
        np.testing.assert_allclose(intensity, [1000.0, 1001.0])

    # ---- loading them all ----------------------------------------------

    def test_020_load_builds_the_matrix(self):
        raman = self.initialized()

        self.assertEqual(raman.shape, (self.expected_spectra, self.POINTS))
        self.assertEqual(len(raman), self.expected_spectra)
        self.assertEqual(raman.X.dtype, np.float64)
        self.assertEqual(raman.axis.shape, (self.POINTS,))
        self.assertFalse(np.isnan(raman.X).any())

    def test_021_every_row_is_described_by_its_own_metadata(self):
        """
        The point of the whole class: row i of X belongs to row i of meta.

        Each spectrum here was given an intensity that depends on the mouse, so
        a mismatch between the two shows up as a wrong number rather than as
        something that merely looks plausible.
        """
        raman = self.initialized()

        self.assertEqual(raman.X.shape[0], len(raman.meta))
        for row, (index, metadata) in enumerate(raman.meta.iterrows()):
            expected = 1000.0 + 100 * int(metadata['sample'])
            self.assertAlmostEqual(raman.X[row, 0], expected,
                                   msg=f"row {row} does not hold the spectrum of {index}")

    def test_022_load_returns_self(self):
        self.datafiles.initialize()
        raman = RamanData(self.datafiles)
        self.assertIs(raman.initialize(verbose=False), raman)

    def test_023_load_without_datafiles(self):
        with self.assertRaises(ValueError):
            RamanData().initialize()

    def test_024_load_only_a_part(self):
        self.datafiles.initialize()
        mask = self.datafiles.get_mask({'sample': 1})
        raman = RamanData(self.datafiles).initialize(mask=mask, verbose=False)

        self.assertEqual(len(raman), 10)
        self.assertTrue((raman.meta['sample'] == 1).all())

    def test_025_a_spectrum_of_another_length_is_left_out(self):
        """A measurement made with another instrument cannot be a row here."""
        write_spectrum_file(self.root / "sample1" / "zone1" / "sample1_dose0_zone1_other_instrument.txt",
                            points=self.POINTS * 2)
        raman = self.initialized()

        self.assertEqual(len(raman), self.expected_spectra)
        self.assertEqual(raman.X.shape[1], self.POINTS)

    def test_026_an_empty_file_is_left_out(self):
        note = (self.root / "sample1" / "zone1" / "note about sample1.txt")
        note.write_text("pas un spectre\n", encoding="latin-1")

        self.assertEqual(len(self.initialized()), self.expected_spectra)

    def test_027_two_different_axes_stop_everything(self):
        """
        Same number of points, different wavelengths: stacking them would put
        one wavelength on top of another and every result would be wrong.
        """
        write_spectrum_file(self.root / "sample1" / "zone1" / "sample1_dose0_zone1_other_axis.txt",
                            points=self.POINTS, first=999.0)
        self.datafiles.initialize()

        with self.assertRaises(ValueError):
            RamanData(self.datafiles).initialize(verbose=False)

    # ---- what went in and what came out ---------------------------------

    def test_028_report_counts_everything(self):
        raman = self.initialized()
        counts = raman.report(verbose=False)

        self.assertEqual(counts['offered'], self.expected_spectra)
        self.assertEqual(counts['kept'], self.expected_spectra)
        self.assertEqual(counts['rejected'], 0)
        self.assertEqual(counts['points'], self.POINTS)

    def test_029_report_names_what_was_left_out(self):
        """The whole point: the missing files can be named, not merely counted."""
        folder = self.root / "sample1" / "zone1"
        empty = folder / "note about sample1.txt"
        empty.write_text("pas un spectre\n", encoding="latin-1")
        write_spectrum_file(folder / "sample1_dose0_zone1_other_instrument.txt",
                            points=self.POINTS * 2)

        raman = self.initialized()
        counts = raman.report(verbose=False)

        self.assertEqual(counts['offered'], self.expected_spectra + 2)
        self.assertEqual(counts['kept'], self.expected_spectra)
        self.assertEqual(counts['rejected'], 2)
        self.assertEqual(counts['offered'], counts['kept'] + counts['rejected'])

        # 0 points means a file that holds no measurement at all
        self.assertEqual(counts['lengths'], {0: 1, self.POINTS * 2: 1})
        self.assertEqual(sorted(Path(name).name for name in raman.rejected),
                         ["note about sample1.txt", "sample1_dose0_zone1_other_instrument.txt"])

    def test_02a_report_before_initializing(self):
        """Asking before reading says nothing rather than raising."""
        counts = RamanData(self.datafiles).report(verbose=False)
        self.assertEqual(counts['offered'], 0)
        self.assertEqual(counts['kept'], 0)
        self.assertEqual(counts['rejected'], 0)

    def test_02b_report_survives_being_written_out(self):
        """What was left out matters more later, not less: it has to be kept."""
        folder = self.root / "sample1" / "zone1"
        (folder / "note about sample1.txt").write_text("pas un spectre\n", encoding="latin-1")

        raman = self.initialized()
        path = Path(self.temporary.name) / "with-report"
        raman.save(path)

        again = RamanData.read(path)
        self.assertEqual(again.report(verbose=False), raman.report(verbose=False))

    # ---- handing the data to scikit-learn -------------------------------

    def test_030_training_set(self):
        raman = self.initialized()
        X, y = raman.training_set('dose')

        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(y.dtype, np.float64)          # not 'object', which sklearn refuses
        self.assertEqual(sorted(set(y)), [0.0, 45.0])

    def test_031_training_set_drops_the_rows_without_a_label(self):
        """X and y have to lose the same rows, or the labels end up shifted."""
        raman = self.initialized()
        raman.meta = raman.meta.copy()
        raman.meta.iloc[0, raman.meta.columns.get_loc('dose')] = None

        X, y = raman.training_set('dose')
        self.assertEqual(X.shape[0], len(raman) - 1)
        self.assertEqual(y.shape[0], len(raman) - 1)
        np.testing.assert_array_equal(X[0], raman.X[1])

    def test_032_training_set_can_keep_them(self):
        raman = self.initialized()
        X, y = raman.training_set('dose', dropna=False)
        self.assertEqual(X.shape[0], len(raman))

    def test_033_training_set_before_initializing(self):
        with self.assertRaises(ValueError):
            RamanData(self.datafiles).training_set('dose')

    def test_034_groups(self):
        raman = self.initialized()
        groups = raman.groups('sample')

        self.assertEqual(len(groups), len(raman))
        self.assertEqual(sorted(set(groups)), [1, 2])

    def test_035_groups_before_initializing(self):
        with self.assertRaises(ValueError):
            RamanData(self.datafiles).groups('sample')

    # ---- averaging the repetitions --------------------------------------

    def test_036_the_repetitions_are_averaged_together(self):
        """
        The five files of one zone say exactly the same thing about themselves,
        so the default fingerprint already puts them together.
        """
        raman = self.initialized()
        mean = raman.averaged(verbose=False)

        self.assertEqual(len(mean), 4)                       # 2 samples x 2 zones
        self.assertEqual(mean.shape, (4, self.POINTS))
        self.assertTrue((mean.meta['n_averaged'] == 5).all())

    def test_037_not_one_spectrum_is_lost(self):
        """
        Every spectrum must end up in exactly one group.

        A missing value in a fingerprint column is the way to lose spectra
        without being told: pandas leaves those rows out of a grouping unless it
        is asked not to. The extra file below has no zone in its name, so its
        'zone' is empty, and the total is what proves it survived.
        """
        write_spectrum_file(self.root / "sample1_dose0_elsewhere.txt", points=self.POINTS)
        raman = self.initialized()
        mean = raman.averaged(verbose=False)

        self.assertEqual(len(raman), self.expected_spectra + 1)
        self.assertEqual(int(mean.meta['n_averaged'].sum()), len(raman))
        self.assertTrue(mean.meta['zone'].isna().any())

    def test_038_the_average_is_the_average_of_the_right_rows(self):
        """
        The same check as test_021, on the averaged table: each spectrum was
        given an intensity that depends on its sample, so a row of X paired with
        the wrong row of meta shows up as a wrong number.
        """
        raman = self.initialized()
        mean = raman.averaged(verbose=False)

        for row, (label, metadata) in enumerate(mean.meta.iterrows()):
            expected = 1000.0 + 100 * int(metadata['sample'])
            self.assertAlmostEqual(mean.X[row, 0], expected,
                                   msg=f"row {row} does not hold the average of {label}")

    def test_039_ignoring_a_column_merges_more(self):
        """Leaving 'zone' out joins the two zones of a sample into one row."""
        raman = self.initialized()
        mean = raman.averaged(ignore=['zone'], verbose=False)

        self.assertEqual(len(mean), 2)
        self.assertTrue((mean.meta['n_averaged'] == 10).all())

    def test_03a_on_gives_the_fingerprint_directly(self):
        raman = self.initialized()
        mean = raman.averaged(on=['sample'], verbose=False)

        self.assertEqual(len(mean), 2)
        self.assertEqual(sorted(mean.meta['sample']), [1, 2])
        self.assertTrue((mean.meta['n_averaged'] == 10).all())

    def test_03b_a_column_the_group_disagrees_on_is_left_empty(self):
        """
        Grouping by sample alone puts both zones in the same row. There is no
        single zone for that row, so the cell is emptied rather than holding
        whichever zone happened to come first. 'dose' does not move within a
        sample, so it is kept.
        """
        raman = self.initialized()
        mean = raman.averaged(on=['sample'], verbose=False)

        self.assertTrue(mean.meta['zone'].isna().all())
        self.assertEqual(sorted(mean.meta['dose']), [0, 45])

    def test_03c_the_rows_are_labelled_by_their_fingerprint(self):
        raman = self.initialized()
        mean = raman.averaged(on=['sample', 'zone'], verbose=False)

        self.assertTrue(mean.meta.index.is_unique)
        self.assertIn("sample=1 zone=1", list(mean.meta.index))

    def test_03d_an_empty_fingerprint_averages_everything(self):
        raman = self.initialized()
        mean = raman.averaged(on=[], verbose=False)

        self.assertEqual(len(mean), 1)
        self.assertEqual(int(mean.meta['n_averaged'].iloc[0]), len(raman))
        self.assertAlmostEqual(mean.X[0, 0], raman.X[:, 0].mean())

    def test_03e_a_misspelled_column_is_refused(self):
        """A typo must fail rather than quietly change what is averaged."""
        raman = self.initialized()
        with self.assertRaises(ValueError):
            raman.averaged(ignore=['zonne'], verbose=False)
        with self.assertRaises(ValueError):
            raman.averaged(on=['sample', 'zonne'], verbose=False)

    def test_03f_ignore_and_on_cannot_both_be_given(self):
        raman = self.initialized()
        with self.assertRaises(ValueError):
            raman.averaged(ignore=['zone'], on=['sample'], verbose=False)

    def test_03g_averaging_before_initializing(self):
        with self.assertRaises(ValueError):
            RamanData(self.datafiles).averaged()

    def test_03h_the_original_is_left_untouched(self):
        raman = self.initialized()
        before = raman.X.copy()
        raman.averaged(on=['sample'], verbose=False)

        self.assertEqual(len(raman), self.expected_spectra)
        self.assertTrue(np.array_equal(raman.X, before))

    def test_03i_an_average_can_be_saved_and_read_back(self):
        raman = self.initialized()
        mean = raman.averaged(on=['sample'], verbose=False)
        path = mean.save(Path(self.temporary.name) / "mean")

        again = RamanData.read(path)
        self.assertTrue(np.array_equal(again.X, mean.X))
        self.assertEqual(list(again.meta.index), list(mean.meta.index))
        self.assertEqual(list(again.meta['n_averaged']), list(mean.meta['n_averaged']))

    # ---- writing and reading back ---------------------------------------

    def test_040_save_and_read(self):
        raman = self.initialized()
        path = Path(self.temporary.name) / "saved"
        raman.save(path)

        again = RamanData.read(path)
        np.testing.assert_array_equal(again.X, raman.X)
        np.testing.assert_array_equal(again.axis, raman.axis)
        self.assertTrue(again.meta.index.equals(raman.meta.index))

    def test_041_read_notices_a_mismatch(self):
        """
        Regenerating one of the two files and not the other must be caught.

        Otherwise the spectra keep being described by the metadata of other
        spectra, and nothing anywhere says so.
        """
        raman = self.initialized()
        path = Path(self.temporary.name) / "mismatched"
        raman.save(path)

        shuffled = raman.meta.iloc[::-1]
        shuffled.to_pickle(path.with_suffix(".pkl"))

        with self.assertRaises(ValueError):
            RamanData.read(path)

    def test_042_save_before_initializing(self):
        with self.assertRaises(ValueError):
            RamanData(self.datafiles).save(Path(self.temporary.name) / "nothing")


if __name__ == "__main__":
    unittest.main()
