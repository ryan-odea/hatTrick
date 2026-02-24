import glob as _glob
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
from natsort import natsorted

try:
    from bitshuffle.h5 import H5_COMPRESS_LZ4, H5FILTER

    HAS_BITSHUFFLE = True
except ImportError:
    HAS_BITSHUFFLE = False
    H5_COMPRESS_LZ4 = None
    H5FILTER = None


# ---------------------------------------------------------------------------
# Prime / S-matrix helpers
# ---------------------------------------------------------------------------


def isprime(n: int) -> bool:
    """Return True if *n* is a prime number.

    :param n: Integer to test for primality
    :returns: True if n is prime, False otherwise
    """
    if n < 2:
        return False
    for x in range(2, int(n**0.5) + 1):
        if n % x == 0:
            return False
    return True


def ishift(alist: list, n: int) -> list:
    """Left-rotate *alist* by *n* positions in place and return it.

    :param alist: List to rotate
    :param n: Number of positions to rotate left
    :returns: The rotated list (same object as input)
    """
    for i in range(n):
        tmp = alist.pop(0)
        alist.append(tmp)
    return alist


def generate_s_matrix(n: int) -> np.ndarray:
    """Generate an n x n S-matrix via the quadratic residues construction.

    :param n: Order of the matrix; must be prime and satisfy n % 4 == 3
    :returns: The n x n binary S-matrix
    """
    if not isprime(n):
        raise ValueError(f"n={n} must be prime for quadratic residues method")
    if n % 4 != 3:
        raise ValueError(f"n={n} must satisfy n == 3 (mod 4) for this construction")

    m = range(0, n)
    Srow = [0 for _ in m]

    for i in range(0, (n - 1) // 2):
        Srow[(i + 1) * (i + 1) % n] = 1
    Srow[0] = 1

    S = [[0 for _ in m] for __ in m]
    rowcopy = Srow.copy()
    for i in m:
        for j in m:
            S[i][j] = rowcopy[j]
        rowcopy = ishift(rowcopy, 1)

    return np.array(S)


def compute_hadamard_inverse(S: np.ndarray) -> np.ndarray:
    """Compute the inverse of an S-matrix using the Hadamard formula.

    For an S-matrix of order n the inverse is::

        S^-1[i,j] = 2 * (2 * S^T[i,j] - 1) / (n + 1)

    :param S: Square binary S-matrix of shape ``(n, n)``
    :returns: The inverse matrix of shape ``(n, n)``
    """
    n = S.shape[0]
    JN = np.ones((n, n))
    ST = S.T
    Sinv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Sinv[i][j] = 2.0 * (2.0 * ST[i][j] - JN[i][j]) / (n + 1)
    return Sinv


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_hadamard(n_merged_frames: int) -> None:
    """Validate that *n_merged_frames* satisfies the Hadamard prime constraint.

    :param n_merged_frames: Block size; must be a positive prime satisfying n % 4 == 3
    """
    if n_merged_frames <= 0:
        raise ValueError("n_merged_frames must be a positive integer.")
    if not isprime(n_merged_frames):
        raise ValueError(f"n_merged_frames={n_merged_frames} must be prime for Hadamard encoding")
    if n_merged_frames % 4 != 3:
        raise ValueError(f"n_merged_frames={n_merged_frames} must satisfy n == 3 (mod 4)")


# ---------------------------------------------------------------------------
# File-list / glob helpers
# ---------------------------------------------------------------------------


def resolve_file_list(source: str) -> List[str]:
    """Resolve a file-list path or glob pattern to an ordered list of paths.

    When the number of files is large, shell glob expansion can exceed
    ``ARG_MAX`` and fail. This function performs expansion inside Python so
    that no shell limit applies.

    The *source* argument is interpreted as follows:

    * If it is an existing regular file whose extension is **not** ``.h5`` or
      ``.hdf5``, it is treated as a **list file**: one HDF5 path per line,
      with ``#``-prefixed lines and blank lines ignored.
    * Otherwise it is treated as a **glob pattern** and expanded with
      :func:`glob.glob`.

    :param source: Either a glob pattern (e.g. ``"data/run_*.h5"``) or the path to a plain-text manifest file containing one HDF5 path per line
    :returns: Sorted list of resolved file paths
    :raises ValueError: If no files are found after resolution
    """
    if os.path.isfile(source) and not source.endswith(".h5") and not source.endswith(".hdf5"):
        paths = []
        with open(source, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(line)
        if not paths:
            raise ValueError(f"List file '{source}' contains no file paths.")
        return paths

    paths = natsorted(_glob.glob(source))
    if not paths:
        raise ValueError(f"No files matched glob pattern '{source}'.")
    return paths


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _open_file(
    file_name: str, data_location: str, data_name: str
) -> Tuple[h5py.File, h5py.Dataset]:
    """Open an HDF5 file and return the file handle and dataset.

    :param file_name: Path to the HDF5 file
    :param data_location: HDF5 group path (e.g., "entry/data")
    :param data_name: Dataset name inside the data_location group
    :returns: Tuple of (h5py.File, h5py.Dataset) - caller is responsible for closing the file handle
    :raises IOError: If the file cannot be opened
    :raises KeyError: If the dataset is not found in the file
    """
    try:
        data_file = h5py.File(file_name, "r")
    except Exception as e:
        raise IOError(f"Could not open file {file_name}: {e}")

    data_path = f"{data_location}/{data_name}"
    if data_path not in data_file:
        raise KeyError(f"Dataset {data_path} not found in file {file_name}.")

    return data_file, data_file[data_path]


def _frame_iterator(
    file_paths: List[str],
    data_location: str,
    data_name: str,
    start_index: int = 0,
) -> Iterator[Tuple[np.ndarray, "np.dtype", str]]:
    """Yield individual frames across an ordered sequence of HDF5 files.

    At most one HDF5 file is open at a time. Combined with the block-at-a-time
    accumulation in :func:`continuous_hadamard_encode`, at most one block's
    worth of frame data is in memory simultaneously.

    :param file_paths: Ordered list of HDF5 file paths to stream through
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param start_index: Global frame index at which to begin yielding (0-based, counting from the very first frame of *file_paths[0]*)
    :yields: Tuple of ``(frame_array, dtype, file_path)`` for each frame at or after *start_index*
    """
    global_idx = 0
    for file_path in file_paths:
        fh, dset = _open_file(file_path, data_location, data_name)
        try:
            n = len(dset)
            for local_idx in range(n):
                if global_idx >= start_index:
                    yield dset[local_idx], dset.dtype, file_path
                global_idx += 1
        finally:
            fh.close()


# ---------------------------------------------------------------------------
# Incremental HDF5 writer
# ---------------------------------------------------------------------------


def _dominant_stem(file_counts: Counter) -> str:
    """Return the stem of the file that contributed the most frames to a block.

    :param file_counts: Counter mapping file paths to frame counts
    :returns: The stem (filename without extension) of the dominant file
    """
    dominant_path = file_counts.most_common(1)[0][0]
    return Path(dominant_path).stem


class _IncrementalWriter:
    """Manages per-pattern HDF5 output files and appends bunches incrementally.

    Output files are named ``{output_dir}/{dominant_stem}_{binary_tag}.h5``
    where *binary_tag* is the S-matrix row for that pattern.  When the dominant
    file changes, previously open files are closed and new ones are opened.
    When the dominant file is the same as the previous bunch, data is appended
    to the existing resizable datasets.

    :param output_dir: Directory in which to write output files
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param S_matrix: The S-matrix used for encoding
    """

    def __init__(
        self,
        output_dir: str,
        data_location: str,
        data_name: str,
        S_matrix: np.ndarray,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_location = data_location
        self.data_name = data_name
        self.S_matrix = S_matrix
        self.n_patterns = S_matrix.shape[0]

        self._binary_tags = []
        for row in S_matrix:
            self._binary_tags.append("".join(str(int(x)) for x in row))

        # Map: dominant_stem -> list of n open h5py.File handles
        self._open_files: Dict[str, List[h5py.File]] = {}
        self._compress_kwargs = {}
        if HAS_BITSHUFFLE:
            self._compress_kwargs = dict(
                compression=H5FILTER,
                compression_opts=(0, H5_COMPRESS_LZ4),
            )

    def append_bunch(
        self,
        encoded: np.ndarray,
        dtype,
        dominant_stem: str,
    ) -> None:
        """Append one encoded bunch (n_patterns, H, W) to the output files.

        :param encoded: Array of shape ``(n_patterns, H, W)``
        :param dtype: HDF5 dataset dtype
        :param dominant_stem: Stem of the dominant input file for this bunch
        """
        if dominant_stem not in self._open_files:
            self._open_files[dominant_stem] = self._open_pattern_files(
                dominant_stem, encoded.shape[1:], dtype
            )

        handles = self._open_files[dominant_stem]
        for pattern_idx in range(self.n_patterns):
            fh = handles[pattern_idx]
            dset = fh[f"{self.data_location}/{self.data_name}"]
            cur_len = dset.shape[0]
            dset.resize(cur_len + 1, axis=0)
            dset[cur_len] = encoded[pattern_idx]

    def _open_pattern_files(
        self, dominant_stem: str, frame_shape: tuple, dtype
    ) -> List[h5py.File]:
        """Create and return n_patterns new HDF5 files for a dominant stem."""
        handles = []
        for pattern_idx in range(self.n_patterns):
            tag = self._binary_tags[pattern_idx]
            path = self.output_dir / f"{dominant_stem}_{tag}.h5"
            fh = h5py.File(str(path), "w")
            grp = fh.create_group(self.data_location)
            dset = grp.create_dataset(
                self.data_name,
                shape=(0, *frame_shape),
                maxshape=(None, *frame_shape),
                chunks=(1, *frame_shape),
                dtype=dtype,
                **self._compress_kwargs,
            )
            grp.attrs["encoding_type"] = "hadamard"
            grp.attrs["pattern_index"] = pattern_idx
            grp.attrs["n_patterns"] = self.n_patterns
            grp.attrs["s_matrix_row"] = self.S_matrix[pattern_idx]
            handles.append(fh)
        return handles

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for handles in self._open_files.values():
            for fh in handles:
                fh.close()
        self._open_files.clear()

    def written_stems(self) -> List[str]:
        """Return the list of dominant stems that were written."""
        return list(self._open_files.keys())


# ---------------------------------------------------------------------------
# Hadamard encode functions
# ---------------------------------------------------------------------------


def single_file_hadamard_encode(
    file_paths: List[str],
    n_merged_frames: int,
    data_location: str,
    data_name: str,
    output_dir: str,
    start_index: int = 0,
) -> None:
    """Hadamard-encode frames from multiple files, processing each file independently.

    Each file is processed separately, and blocks do NOT span file boundaries.
    Within each file, blocks of *n_merged_frames* frames are consumed and
    encoded with the S-matrix.  Output files are written incrementally and
    named after the input file stem (e.g. ``input_000_110.h5``).

    :param file_paths: Ordered list of HDF5 files to process independently
    :param n_merged_frames: S-matrix order; must be prime and satisfy n % 4 == 3
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param output_dir: Directory for output files
    :param start_index: Global frame index (0-based) at which to begin forming blocks in the first file
    :raises ValueError: If no complete blocks can be formed
    """
    _validate_hadamard(n_merged_frames)
    S = generate_s_matrix(n_merged_frames)

    writer = _IncrementalWriter(output_dir, data_location, data_name, S)
    dtype = None
    frame_shape = None
    global_frame_idx = 0
    n_bunches_total = 0

    try:
        for file_idx, file_path in enumerate(file_paths):
            fh, dset = _open_file(file_path, data_location, data_name)
            try:
                n_frames = len(dset)
                file_stem = Path(file_path).stem

                if dtype is None:
                    dtype = dset.dtype
                    if n_frames > 0:
                        frame_shape = dset[0].shape

                local_start = 0
                if file_idx == 0 and start_index > 0:
                    local_start = start_index

                if local_start >= n_frames:
                    global_frame_idx += n_frames
                    continue

                block: List[np.ndarray] = []

                for local_idx in range(local_start, n_frames):
                    frame = dset[local_idx]
                    if frame_shape is None:
                        frame_shape = frame.shape

                    block.append(frame)

                    if len(block) == n_merged_frames:
                        encoded = np.zeros((n_merged_frames, *frame_shape), dtype=dtype)
                        for pattern_idx in range(n_merged_frames):
                            for frame_idx in range(n_merged_frames):
                                if S[pattern_idx, frame_idx] == 1:
                                    encoded[pattern_idx] += block[frame_idx]

                        writer.append_bunch(encoded, dtype, file_stem)
                        n_bunches_total += 1
                        block = []

                global_frame_idx += n_frames

            finally:
                fh.close()

        if n_bunches_total == 0:
            raise ValueError("No complete blocks found; check start_index and file contents.")
    finally:
        writer.close()


def continuous_hadamard_encode(
    file_paths: List[str],
    n_merged_frames: int,
    data_location: str,
    data_name: str,
    output_dir: str,
    start_index: int = 0,
) -> None:
    """Hadamard-encode frames streamed continuously across multiple HDF5 files.

    Blocks of *n_merged_frames* frames are consumed from the ordered sequence
    of *file_paths* and encoded with the S-matrix.  Blocks may span file
    boundaries.  Output files are written incrementally and named after the
    dominant input file (the file contributing the most frames to each block),
    e.g. ``input_001_110.h5``.

    :param file_paths: Ordered list of HDF5 files to stream through
    :param n_merged_frames: S-matrix order; must be prime and satisfy n % 4 == 3
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param output_dir: Directory for output files
    :param start_index: Global frame index (0-based, counting from the first frame of *file_paths[0]*) at which to begin forming blocks
    :raises ValueError: If *start_index* is beyond all available frames, or if no complete blocks can be formed
    """
    _validate_hadamard(n_merged_frames)
    S = generate_s_matrix(n_merged_frames)

    # Peek at first frame to get dtype/shape.
    peek = _frame_iterator(file_paths, data_location, data_name, start_index)
    try:
        first_frame, dtype, _ = next(peek)
    except StopIteration:
        raise ValueError("No frames available at or after start_index.")
    frame_shape = first_frame.shape
    del first_frame

    writer = _IncrementalWriter(output_dir, data_location, data_name, S)
    block: List[np.ndarray] = []
    block_file_counts: Counter = Counter()
    n_bunches_total = 0

    try:
        for frame, _, file_path in _frame_iterator(
            file_paths, data_location, data_name, start_index
        ):
            block.append(frame)
            block_file_counts[file_path] += 1

            if len(block) == n_merged_frames:
                encoded = np.zeros((n_merged_frames, *frame_shape), dtype=dtype)
                for pattern_idx in range(n_merged_frames):
                    for frame_idx in range(n_merged_frames):
                        if S[pattern_idx, frame_idx] == 1:
                            encoded[pattern_idx] += block[frame_idx]

                stem = _dominant_stem(block_file_counts)
                writer.append_bunch(encoded, dtype, stem)
                n_bunches_total += 1
                block = []
                block_file_counts = Counter()

        if n_bunches_total == 0:
            raise ValueError("No complete blocks found; check start_index and file contents.")
    finally:
        writer.close()
