import glob as _glob
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterator, List, Set, Tuple

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


def isprime(n: int) -> bool:
    if n < 2:
        return False
    for x in range(2, int(n**0.5) + 1):
        if n % x == 0:
            return False
    return True


def ishift(alist: list, n: int) -> list:
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


def hadamard_decode(values: np.ndarray, sigmas: np.ndarray, Sinv: np.ndarray):
    """Apply the inverse S-matrix to encoded values; sigmas propagate in quadrature.

    ``sigma_i = sqrt(sum_j Sinv[i, j]^2 * sigma_j^2)``

    :param values: Encoded values of shape ``(m, n)``; column j is pattern j
    :param sigmas: Encoded sigmas of the same shape
    :param Sinv: Inverse S-matrix of shape ``(n, n)``
    :returns: ``(decoded_values, decoded_sigmas)`` of shape ``(m, n)``
    """
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    return values @ Sinv.T, np.sqrt(sigmas**2 @ (Sinv**2).T)


def _validate_hadamard(n_merged_frames: int) -> None:
    if n_merged_frames <= 0:
        raise ValueError("n_merged_frames must be a positive integer.")
    if not isprime(n_merged_frames):
        raise ValueError(f"n_merged_frames={n_merged_frames} must be prime for Hadamard encoding")
    if n_merged_frames % 4 != 3:
        raise ValueError(f"n_merged_frames={n_merged_frames} must satisfy n == 3 (mod 4)")


def resolve_file_list(source: str) -> List[str]:
    """Resolve a glob pattern or plain-text list file to an ordered list of paths.

    An existing file not ending in ``.h5``/``.hdf5`` is read as one path per
    line (``#`` lines ignored); anything else is expanded as a glob.

    :param source: Glob pattern or list-file path
    :returns: Naturally sorted file paths
    :raises ValueError: If nothing resolves
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


def _open_file(
    file_name: str, data_location: str, data_name: str
) -> Tuple[h5py.File, h5py.Dataset]:
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


def _dominant_stem(file_counts: Counter) -> str:
    dominant_path = file_counts.most_common(1)[0][0]
    return Path(dominant_path).stem


class _IncrementalWriter:

    def __init__(
        self,
        output_dir: str,
        data_location: str,
        data_name: str,
        S_matrix: np.ndarray,
        max_open_stems: int = 64,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_location = data_location
        self.data_name = data_name
        self.S_matrix = S_matrix
        self.n_patterns = S_matrix.shape[0]
        self.max_open_stems = max_open_stems

        self._binary_tags = []
        for row in S_matrix:
            self._binary_tags.append("".join(str(int(x)) for x in row))

        # LRU: stem -> open h5py.File handles
        self._open_files: OrderedDict[str, List[h5py.File]] = OrderedDict()
        # stems already on disk, reopened in append mode
        self._created_stems: Set[str] = set()
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
        if dominant_stem not in self._open_files:
            # evict least recently used stems
            while len(self._open_files) >= self.max_open_stems:
                evict_stem, evict_handles = self._open_files.popitem(last=False)
                for fh in evict_handles:
                    fh.close()

            if dominant_stem in self._created_stems:
                self._open_files[dominant_stem] = self._reopen_pattern_files(
                    dominant_stem
                )
            else:
                self._open_files[dominant_stem] = self._open_pattern_files(
                    dominant_stem, encoded.shape[1:], dtype
                )
                self._created_stems.add(dominant_stem)
        else:
            # mark most recently used
            self._open_files.move_to_end(dominant_stem)

        handles = self._open_files[dominant_stem]
        for pattern_idx in range(self.n_patterns):
            fh = handles[pattern_idx]
            dset = fh[f"{self.data_location}/{self.data_name}"]
            cur_len = dset.shape[0]
            dset.resize(cur_len + 1, axis=0)
            try:
                dset[cur_len] = encoded[pattern_idx]
            except Exception as exc:
                raise type(exc)(
                    f"Failed writing pattern {pattern_idx} to "
                    f"{fh.filename!r}: dset.shape={dset.shape}, "
                    f"encoded[pattern_idx].shape={encoded[pattern_idx].shape}, "
                    f"encoded.dtype={encoded.dtype}, "
                    f"dset.dtype={dset.dtype}, cur_len={cur_len}"
                ) from exc

    def _open_pattern_files(
        self, dominant_stem: str, frame_shape: tuple, dtype
    ) -> List[h5py.File]:
        handles = []
        for pattern_idx in range(self.n_patterns):
            tag = self._binary_tags[pattern_idx]
            path = self.output_dir / f"{dominant_stem}_{tag}.h5"
            fh = h5py.File(str(path), "w")
            grp = fh.create_group(self.data_location)
            grp.create_dataset(
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

    def _reopen_pattern_files(self, dominant_stem: str) -> List[h5py.File]:
        handles = []
        for pattern_idx in range(self.n_patterns):
            tag = self._binary_tags[pattern_idx]
            path = self.output_dir / f"{dominant_stem}_{tag}.h5"
            fh = h5py.File(str(path), "a")
            handles.append(fh)
        return handles

    def close(self) -> None:
        for handles in self._open_files.values():
            for fh in handles:
                fh.close()
        self._open_files.clear()

    def written_stems(self) -> List[str]:
        return list(self._created_stems)


def single_file_hadamard_encode(
    file_paths: List[str],
    n_merged_frames: int,
    data_location: str,
    data_name: str,
    output_dir: str,
    start_index: int = 0,
) -> None:
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
    _validate_hadamard(n_merged_frames)
    S = generate_s_matrix(n_merged_frames)

    # peek first frame for dtype and shape
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
