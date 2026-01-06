import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
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
    if not isprime(n):
        raise ValueError(f"n={n} must be prime for quadratic residues method")
    if n % 4 != 3:
        raise ValueError(f"n={n} must satisfy n ≡ 3 (mod 4) for this construction")

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


def _validate(
    file_name: str, n_merged_frames: int, skip_frames: Optional[List[int]] = None
) -> None:
    if not file_name or not os.path.isfile(file_name):
        raise ValueError(f"Input file {file_name} does not exist.")

    if n_merged_frames <= 0:
        raise ValueError("n_merged_frames must be a positive integer.")

    if not isprime(n_merged_frames):
        raise ValueError(
            f"n_merged_frames={n_merged_frames} must be prime for Hadamard encoding"
        )
    if n_merged_frames % 4 != 3:
        raise ValueError(
            f"n_merged_frames={n_merged_frames} must satisfy n ≡ 3 (mod 4)"
        )

    if skip_frames is not None:
        if not isinstance(skip_frames, list):
            raise ValueError("skip_frames must be a list of integers.")
        for skip_idx in skip_frames:
            if not isinstance(skip_idx, int) or skip_idx < 0:
                raise ValueError(f"Invalid skip index: {skip_idx}")
            if skip_idx >= n_merged_frames:
                raise ValueError(
                    f"Skip frame index {skip_idx} not in valid range [0, {n_merged_frames - 1}]"
                )
        if len(skip_frames) >= n_merged_frames:
            raise ValueError(f"Cannot skip all {n_merged_frames} frames in each group.")


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

    data = data_file[data_path]
    return data_file, data


def _create_merge_indices(n_frames: int, n_merged_frames: int) -> List[int]:
    indices = []
    for i in range(0, n_frames, n_merged_frames):
        if n_frames - i >= n_merged_frames:
            indices.append(i)
    return indices


def _hadamard_encode_chunk_mp(args: Tuple) -> Tuple[int, np.ndarray]:
    start_idx, data_subset, S, dtype = args
    n_merged_frames = S.shape[0]
    frame_shape = data_subset.shape[1:]

    encoded = np.zeros((n_merged_frames, *frame_shape), dtype=dtype)

    for pattern_idx in range(n_merged_frames):
        for frame_idx in range(n_merged_frames):
            if S[pattern_idx, frame_idx] == 1:
                encoded[pattern_idx] += data_subset[frame_idx]

    return start_idx, encoded


def _hadamard_encode_chunk_sq(
    data_array: np.ndarray,
    n_frames: int,
    n_merged_frames: int,
    frame_shape: Tuple[int],
    S: np.ndarray,
    dtype=None,
) -> np.ndarray:
    merge_indices = _create_merge_indices(n_frames, n_merged_frames)
    n_bunches = len(merge_indices)

    encoded_data = np.zeros((n_merged_frames, n_bunches, *frame_shape), dtype=dtype)

    for bunch_idx, start_idx in enumerate(merge_indices):
        for pattern_idx in range(n_merged_frames):
            frame_encoded = np.zeros(frame_shape, dtype=dtype)
            for frame_idx in range(n_merged_frames):
                if S[pattern_idx, frame_idx] == 1:
                    frame_encoded += data_array[start_idx + frame_idx]
            encoded_data[pattern_idx, bunch_idx] = frame_encoded

    return encoded_data


def _rolling_merge_sq(
    data_array: np.ndarray,
    n_frames: int,
    n_merged_frames: int,
    frame_shape: Tuple[int],
    skip_pattern: Optional[List[int]] = None,
    dtype=None,
) -> np.ndarray:
    merge_indices = _create_merge_indices(n_frames, n_merged_frames)
    merged_data = np.zeros((len(merge_indices), *frame_shape), dtype=dtype)

    skip_set = set(skip_pattern) if skip_pattern else set()

    for i, start_idx in enumerate(merge_indices):
        frame_merged = np.zeros(frame_shape, dtype=dtype)
        for j in range(n_merged_frames):
            if j not in skip_set:
                frame_merged += data_array[start_idx + j]
        merged_data[i] = frame_merged

    return merged_data


def _write_output(
    output_file: str,
    data_location: str,
    data_name: str,
    merged_data: np.ndarray,
    dtype,
    compression: Optional[Tuple[str, Tuple]] = None,
    S_matrix: Optional[np.ndarray] = None,
) -> None:
    compression_opts = compression[1] if compression else (0, H5_COMPRESS_LZ4)

    if merged_data.ndim == 4:
        n_patterns = merged_data.shape[0]
        base_path = output_file.rsplit(".h5", 1)[0]

        for pattern_idx in range(n_patterns):
            if S_matrix is not None:
                binary_tag = "".join(str(int(x)) for x in S_matrix[pattern_idx])
                pattern_file = f"{base_path}_{binary_tag}.h5"
            else:
                pattern_file = f"{base_path}_pattern{pattern_idx}.h5"

            with h5py.File(pattern_file, "w") as f:
                grp = f.create_group(data_location)
                dset = grp.create_dataset(
                    data_name,
                    merged_data[pattern_idx].shape,
                    chunks=(1, merged_data.shape[2], merged_data.shape[3]),
                    compression=H5FILTER,
                    compression_opts=compression_opts,
                    dtype=dtype,
                )
                dset[:] = merged_data[pattern_idx]

                grp.attrs["encoding_type"] = "hadamard"
                grp.attrs["pattern_index"] = pattern_idx
                grp.attrs["n_patterns"] = n_patterns
                if S_matrix is not None:
                    grp.attrs["s_matrix_row"] = S_matrix[pattern_idx]

        print(f"\nSaved {n_patterns} encoding patterns to {base_path}_*.h5")
    else:
        with h5py.File(output_file, "w") as f:
            grp = f.create_group(data_location)
            dset = grp.create_dataset(
                data_name,
                merged_data.shape,
                chunks=(1, merged_data.shape[1], merged_data.shape[2]),
                compression=H5FILTER,
                compression_opts=compression_opts,
                dtype=dtype,
            )
            dset[:] = merged_data
