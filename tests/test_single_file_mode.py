"""Tests for single-file (non-continuous) encoding mode."""

from pathlib import Path
from typing import List

import h5py
import numpy as np

from hatTrick._helpers import (
    continuous_hadamard_encode,
    generate_s_matrix,
    single_file_hadamard_encode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_LOCATION = "entry/data"
DATA_NAME = "data"
FRAME_SHAPE = (8, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_h5(path: str, frames: np.ndarray) -> None:
    """Write *frames* (n_frames, H, W) to a minimal HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(DATA_LOCATION)
        grp.create_dataset(DATA_NAME, data=frames)


def _read_h5(path: str) -> np.ndarray:
    """Return the dataset from *path* as an ndarray."""
    with h5py.File(path, "r") as f:
        return f[f"{DATA_LOCATION}/{DATA_NAME}"][:]


def _make_input_files(
    directory: Path,
    frames_per_file: List[int],
    seed: int = 0,
) -> List[str]:
    """Write frames split across multiple HDF5 files and return paths."""
    rng = np.random.default_rng(seed)
    paths = []

    for i, count in enumerate(frames_per_file):
        frames = rng.uniform(10.0, 500.0, (count, *FRAME_SHAPE)).astype(np.float32)
        p = str(directory / f"input_{i:03d}.h5")
        _write_h5(p, frames)
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleFileMode:
    """Test single-file (non-continuous) encoding mode."""

    def test_single_file_no_boundary_crossing(self, tmp_path):
        """Verify single-file mode doesn't allow blocks to cross file boundaries."""
        n = 3
        # Create two files, each with exactly 6 frames (2 complete blocks)
        paths = _make_input_files(tmp_path, [6, 6])

        output_base = str(tmp_path / "encoded.h5")

        # Run single-file encode
        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_file=output_base,
            start_index=0,
        )

        # Should have created 4 blocks total (2 per file, no cross-boundary)
        S = generate_s_matrix(n)
        pattern_file = str(tmp_path / f"encoded_{''.join(str(int(x)) for x in S[0])}.h5")

        encoded = _read_h5(pattern_file)
        # 4 blocks total (2 from each file)
        assert encoded.shape[0] == 4

    def test_single_file_discards_trailing_frames(self, tmp_path):
        """Verify incomplete blocks at end of each file are discarded."""
        n = 3
        # Create two files: first has 5 frames (1 complete + 2 trailing),
        # second has 4 frames (1 complete + 1 trailing)
        paths = _make_input_files(tmp_path, [5, 4])

        output_base = str(tmp_path / "encoded.h5")

        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_file=output_base,
            start_index=0,
        )

        # Should have 2 blocks total (1 from each file)
        S = generate_s_matrix(n)
        pattern_file = str(tmp_path / f"encoded_{''.join(str(int(x)) for x in S[0])}.h5")

        encoded = _read_h5(pattern_file)
        assert encoded.shape[0] == 2

    def test_continuous_vs_single_file_difference(self, tmp_path):
        """Verify continuous and single-file modes produce different results."""
        n = 3
        # Create two files with 4 frames each
        # Continuous: can make 2 complete blocks (0-2, 3-5), spanning boundary
        # Single-file: 1 block per file (0-2 from file1, 0-2 from file2), no boundary cross
        paths = _make_input_files(tmp_path, [4, 4])

        # Single-file mode
        single_output = str(tmp_path / "single.h5")
        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_file=single_output,
            start_index=0,
        )

        # Continuous mode
        continuous_output = str(tmp_path / "continuous.h5")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_file=continuous_output,
            start_index=0,
        )

        S = generate_s_matrix(n)
        pattern_tag = "".join(str(int(x)) for x in S[0])

        single_encoded = _read_h5(str(tmp_path / f"single_{pattern_tag}.h5"))
        continuous_encoded = _read_h5(str(tmp_path / f"continuous_{pattern_tag}.h5"))

        # Both should have 2 blocks
        assert single_encoded.shape[0] == 2
        assert continuous_encoded.shape[0] == 2

        # The first block should be the same (both use frames 0-2 from first file)
        assert np.allclose(single_encoded[0], continuous_encoded[0])

        # The second block should be DIFFERENT:
        # - single-file: frames 0-2 from second file
        # - continuous: frames 3 from first file + 0-1 from second file
        assert not np.allclose(single_encoded[1], continuous_encoded[1])

    def test_start_index_only_affects_first_file(self, tmp_path):
        """Verify start_index only applies to the first file in single-file mode."""
        n = 3
        # Create two files with 6 frames each
        paths = _make_input_files(tmp_path, [6, 6])

        output_base = str(tmp_path / "encoded.h5")

        # Start at frame 3 (should skip first block of first file)
        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_file=output_base,
            start_index=3,
        )

        S = generate_s_matrix(n)
        pattern_file = str(tmp_path / f"encoded_{''.join(str(int(x)) for x in S[0])}.h5")

        encoded = _read_h5(pattern_file)
        # Should have 3 blocks: 1 from first file (skipped first block) + 2 from second file
        assert encoded.shape[0] == 3
