import glob as _glob
import os
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np

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

    paths = sorted(_glob.glob(source))
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
) -> Iterator[Tuple[np.ndarray, "np.dtype"]]:
    """Yield individual frames across an ordered sequence of HDF5 files.

    At most one HDF5 file is open at a time. Combined with the block-at-a-time
    accumulation in :func:`continuous_hadamard_encode`, at most one block's
    worth of frame data is in memory simultaneously.

    :param file_paths: Ordered list of HDF5 file paths to stream through
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param start_index: Global frame index at which to begin yielding (0-based, counting from the very first frame of *file_paths[0]*)
    :yields: Tuple of ``(frame_array, dtype)`` for each frame at or after *start_index*
    """
    global_idx = 0
    for file_path in file_paths:
        fh, dset = _open_file(file_path, data_location, data_name)
        try:
            n = len(dset)
            for local_idx in range(n):
                if global_idx >= start_index:
                    yield dset[local_idx], dset.dtype
                global_idx += 1
        finally:
            fh.close()


def _write_output(
    output_file: str,
    data_location: str,
    data_name: str,
    encoded_data: np.ndarray,
    dtype,
    S_matrix: Optional[np.ndarray] = None,
) -> None:
    """Write Hadamard-encoded data to one HDF5 file per pattern.

    *encoded_data* must be 4-D with shape ``(n_patterns, n_bunches, H, W)``.
    One output file is written per pattern, named using the binary S-matrix
    row when *S_matrix* is provided (e.g. ``merged_110.h5``), otherwise
    ``merged_pattern0.h5``, ``merged_pattern1.h5``, etc.

    :param output_file: Base output path; the .h5 suffix is stripped and replaced with the per-pattern tag
    :param data_location: HDF5 group path (e.g., "entry/data")
    :param data_name: Dataset name within the group
    :param encoded_data: Array of shape (n_patterns, n_bunches, H, W)
    :param dtype: HDF5 dataset dtype
    :param S_matrix: S-matrix used for encoding; used to label output files and stored as an HDF5 attribute
    :raises ValueError: If encoded_data is not 4-D
    """
    if encoded_data.ndim != 4:
        raise ValueError(
            f"encoded_data must be 4-D (n_patterns, n_bunches, H, W), "
            f"got shape {encoded_data.shape}"
        )

    if HAS_BITSHUFFLE:
        compress_kwargs = dict(
            compression=H5FILTER,
            compression_opts=(0, H5_COMPRESS_LZ4),
        )
    else:
        compress_kwargs = {}

    n_patterns = encoded_data.shape[0]
    base_path = output_file.rsplit(".h5", 1)[0]

    for pattern_idx in range(n_patterns):
        if S_matrix is not None:
            binary_tag = "".join(str(int(x)) for x in S_matrix[pattern_idx])
            pattern_file = f"{base_path}_{binary_tag}.h5"
        else:
            pattern_file = f"{base_path}_pattern{pattern_idx}.h5"

        with h5py.File(pattern_file, "w") as f:
            grp = f.create_group(data_location)
            pattern_data = encoded_data[pattern_idx]  # (n_bunches, H, W)
            dset = grp.create_dataset(
                data_name,
                pattern_data.shape,
                chunks=(1, pattern_data.shape[1], pattern_data.shape[2]),
                dtype=dtype,
                **compress_kwargs,
            )
            dset[:] = pattern_data

            grp.attrs["encoding_type"] = "hadamard"
            grp.attrs["pattern_index"] = pattern_idx
            grp.attrs["n_patterns"] = n_patterns
            if S_matrix is not None:
                grp.attrs["s_matrix_row"] = S_matrix[pattern_idx]

    print(f"Saved {n_patterns} encoding patterns to {base_path}_*.h5")


# ---------------------------------------------------------------------------
# Hadamard encode functions
# ---------------------------------------------------------------------------


def single_file_hadamard_encode(
    file_paths: List[str],
    n_merged_frames: int,
    data_location: str,
    data_name: str,
    output_file: str,
    start_index: int = 0,
) -> None:
    """Hadamard-encode frames from multiple files, processing each file independently.

    Each file is processed separately, and blocks do NOT span file boundaries.
    Within each file, blocks of *n_merged_frames* frames are consumed and
    encoded with the S-matrix.

    Memory is bounded: at most one block of *n_merged_frames* frames is held
    in RAM at any time, and at most one HDF5 file is open at once.

    One output HDF5 file is written per encoding pattern, named with the
    binary S-matrix row (e.g. ``merged_110.h5`` for n=3).  Any trailing
    frames in each file that do not form a complete block are discarded with
    a warning.

    :param file_paths: Ordered list of HDF5 files to process independently
    :param n_merged_frames: S-matrix order; must be prime and satisfy n % 4 == 3
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param output_file: Base path for output files (e.g. ``"merged.h5"``)
    :param start_index: Global frame index (0-based) at which to begin forming blocks in the first file
    :raises ValueError: If no complete blocks can be formed
    """
    _validate_hadamard(n_merged_frames)
    S = generate_s_matrix(n_merged_frames)

    all_encoded_bunches: List[np.ndarray] = []
    dtype = None
    frame_shape = None
    global_frame_idx = 0

    for file_idx, file_path in enumerate(file_paths):
        fh, dset = _open_file(file_path, data_location, data_name)
        try:
            n_frames = len(dset)

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
            frames_in_file = 0

            for local_idx in range(local_start, n_frames):
                frame = dset[local_idx]
                if frame_shape is None:
                    frame_shape = frame.shape

                block.append(frame)
                frames_in_file += 1

                if len(block) == n_merged_frames:
                    encoded = np.zeros((n_merged_frames, *frame_shape), dtype=dtype)
                    for pattern_idx in range(n_merged_frames):
                        for frame_idx in range(n_merged_frames):
                            if S[pattern_idx, frame_idx] == 1:
                                encoded[pattern_idx] += block[frame_idx]

                    all_encoded_bunches.append(encoded)
                    block = []

            if block:
                print(
                    f"  File {file_path}: discarding incomplete trailing block "
                    f"of {len(block)} frame(s)."
                )

            global_frame_idx += n_frames

        finally:
            fh.close()

    if not all_encoded_bunches:
        raise ValueError("No complete blocks found; check start_index and file contents.")

    # Stack to (n_merged_frames, n_bunches, H, W).
    stacked = np.stack(all_encoded_bunches, axis=0)  # (n_bunches, n, H, W)
    encoded_data = np.transpose(stacked, (1, 0, 2, 3))  # (n, n_bunches, H, W)

    _write_output(
        output_file=output_file,
        data_location=data_location,
        data_name=data_name,
        encoded_data=encoded_data,
        dtype=dtype,
        S_matrix=S,
    )


def continuous_hadamard_encode(
    file_paths: List[str],
    n_merged_frames: int,
    data_location: str,
    data_name: str,
    output_file: str,
    start_index: int = 0,
) -> None:
    """Hadamard-encode frames streamed continuously across multiple HDF5 files.

    Blocks of *n_merged_frames* frames are consumed from the ordered sequence
    of *file_paths* and encoded with the S-matrix.  Blocks may span file
    boundaries: when the window extends past the end of one file the remaining
    frames are drawn from the next.

    Memory is bounded: at most one block of *n_merged_frames* frames is held
    in RAM at any time, and at most one HDF5 file is open at once.

    One output HDF5 file is written per encoding pattern, named with the
    binary S-matrix row (e.g. ``merged_110.h5`` for n=3).  Any trailing
    frames that do not form a complete block are discarded with a warning.

    :param file_paths: Ordered list of HDF5 files to stream through
    :param n_merged_frames: S-matrix order; must be prime and satisfy n % 4 == 3
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param output_file: Base path for output files (e.g. ``"merged.h5"``)
    :param start_index: Global frame index (0-based, counting from the first frame of *file_paths[0]*) at which to begin forming blocks
    :raises ValueError: If *start_index* is beyond all available frames, or if no complete blocks can be formed
    """
    _validate_hadamard(n_merged_frames)
    S = generate_s_matrix(n_merged_frames)

    peek = _frame_iterator(file_paths, data_location, data_name, start_index)
    try:
        first_frame, dtype = next(peek)
    except StopIteration:
        raise ValueError("No frames available at or after start_index.")
    frame_shape = first_frame.shape
    del first_frame

    encoded_bunches: List[np.ndarray] = []
    block: List[np.ndarray] = []
    bunch_idx = 0

    for frame, _ in _frame_iterator(file_paths, data_location, data_name, start_index):
        block.append(frame)

        if len(block) == n_merged_frames:
            encoded = np.zeros((n_merged_frames, *frame_shape), dtype=dtype)
            for pattern_idx in range(n_merged_frames):
                for frame_idx in range(n_merged_frames):
                    if S[pattern_idx, frame_idx] == 1:
                        encoded[pattern_idx] += block[frame_idx]

            encoded_bunches.append(encoded)
            # TODO
            bunch_idx += 1
            block = []

    if block:
        print(f"  Warning: discarding incomplete trailing block of {len(block)} frame(s).")

    if not encoded_bunches:
        raise ValueError("No complete blocks found; check start_index and file contents.")

    # Stack to (n_merged_frames, n_bunches, H, W).
    stacked = np.stack(encoded_bunches, axis=0)  # (n_bunches, n, H, W)
    encoded_data = np.transpose(stacked, (1, 0, 2, 3))  # (n, n_bunches, H, W)

    _write_output(
        output_file=output_file,
        data_location=data_location,
        data_name=data_name,
        encoded_data=encoded_data,
        dtype=dtype,
        S_matrix=S,
    )
