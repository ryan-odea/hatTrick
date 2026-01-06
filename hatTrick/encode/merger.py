import os
from multiprocessing import Pool, cpu_count
from typing import List, Literal, Optional

import h5py
import numpy as np

from ._helpers import (
    _create_merge_indices,
    _hadamard_encode_chunk_mp,
    _hadamard_encode_chunk_sq,
    _open_file,
    _rolling_merge_sq,
    _validate,
    _write_output,
    generate_s_matrix,
)


class Merger:
    def __init__(
        self,
        file_name: str,
        output_file: str = "merged.h5",
        n_frames: int = 10000,
        n_merged_frames: int = 3,
        skip_pattern: Optional[List[int]] = None,
        data_location: str = "entry/data",
        data_name: str = "data",
        n_workers: Optional[int] = None,
        type: Literal["hadamard", "rolling"] = "hadamard",
    ):
        self.file_name = file_name
        self.output_file = output_file
        self.n_frames = n_frames
        self.n_merged_frames = n_merged_frames
        self.skip_pattern = skip_pattern
        self.data_location = data_location
        self.data_name = data_name
        self.n_workers = n_workers or cpu_count()
        self.type = type
        self.data_file = None
        self.data = None
        self.data_array = None
        self.n_total_frames = None
        self.frame_shape = None
        self.dtype = None
        self.merged_data = None
        self.S_matrix = None

    def validate_inputs(self) -> None:
        if self.type == "hadamard":
            _validate(self.file_name, self.n_merged_frames, None)
        else:
            _validate(self.file_name, self.n_merged_frames, self.skip_pattern)

    def process(self, parallel: bool = False) -> None:
        try:
            self.validate_inputs()
            self._open_and_load()

            if self.type == "hadamard":
                print(f"\nGenerating Hadamard S matrix for n={self.n_merged_frames}...")
                self.S_matrix = generate_s_matrix(self.n_merged_frames)
                print("S matrix:")
                print(self.S_matrix)
                print()

            if parallel and self.n_workers > 1:
                if self.type == "hadamard":
                    self._hadamard_encode_parallel()
                else:
                    self._rolling_merge_parallel()
            else:
                if self.type == "hadamard":
                    self._hadamard_encode_sequential()
                else:
                    self._rolling_merge_sequential()

            _write_output(
                self.output_file,
                self.data_location,
                self.data_name,
                self.merged_data,
                self.dtype,
                None,
                self.S_matrix,
            )
        finally:
            if self.data_file is not None:
                self.data_file.close()

    def _open_and_load(self) -> None:
        self.data_file, self.data = _open_file(
            self.file_name, self.data_location, self.data_name
        )
        self.n_total_frames = len(self.data)
        self.frame_shape = self.data.shape[1:]
        self.dtype = self.data.dtype

        if self.n_total_frames < self.n_frames:
            print(
                f"Warning: Requested {self.n_frames} frames, but only {self.n_total_frames} available. Adjusting n_frames."
            )
            self.n_frames = self.n_total_frames

        self.data_array = self.data[: self.n_frames]

    def _hadamard_encode_parallel(self) -> None:
        chunks = []
        for start_idx in _create_merge_indices(self.n_frames, self.n_merged_frames):
            subset = self.data_array[start_idx : start_idx + self.n_merged_frames]
            chunks.append((start_idx, subset, self.S_matrix, self.dtype))

        with Pool(self.n_workers) as pool:
            results = pool.map(_hadamard_encode_chunk_mp, chunks)

        results.sort(key=lambda x: x[0])
        encoded_arrays = [r[1] for r in results]
        self.merged_data = np.array(encoded_arrays)
        self.merged_data = np.transpose(self.merged_data, (1, 0, 2, 3))

    def _hadamard_encode_sequential(self) -> None:
        self.merged_data = _hadamard_encode_chunk_sq(
            self.data_array,
            self.n_frames,
            self.n_merged_frames,
            self.frame_shape,
            self.S_matrix,
            self.dtype,
        )

    def _rolling_merge_parallel(self) -> None:
        chunks = []
        for start_idx in _create_merge_indices(self.n_frames, self.n_merged_frames):
            subset = self.data_array[start_idx : start_idx + self.n_merged_frames]
            chunks.append(
                (start_idx, subset, self.n_merged_frames, self.skip_pattern, self.dtype)
            )

        with Pool(self.n_workers) as pool:
            results = pool.map(_rolling_merge_mp, chunks)

        results.sort(key=lambda x: x[0])
        self.merged_data = np.array([r[1] for r in results])

    def _rolling_merge_sequential(self) -> None:
        self.merged_data = _rolling_merge_sq(
            self.data_array,
            self.n_frames,
            self.n_merged_frames,
            self.frame_shape,
            self.skip_pattern,
            self.dtype,
        )
