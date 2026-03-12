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


def _find_pattern_files(output_dir: str, dominant_stem: str, n: int) -> List[str]:
    """Return the per-pattern output files in pattern order."""
    S = generate_s_matrix(n)
    out = Path(output_dir)
    paths = []
    for row in S:
        tag = "".join(str(int(x)) for x in row)
        paths.append(str(out / f"{dominant_stem}_{tag}.h5"))
    return paths


def _count_bunches_for_stem(output_dir: str, stem: str, n: int) -> int:
    """Return the number of bunches in the first pattern file for a given stem."""
    pfiles = _find_pattern_files(output_dir, stem, n)
    return _read_h5(pfiles[0]).shape[0]


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

        out_dir = str(tmp_path / "out")

        # Run single-file encode
        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
            start_index=0,
        )

        # Should have created 4 blocks total (2 per file, no cross-boundary)
        # Each file is its own dominant stem
        total_bunches = (
            _count_bunches_for_stem(out_dir, "input_000", n)
            + _count_bunches_for_stem(out_dir, "input_001", n)
        )
        assert total_bunches == 4

    def test_single_file_discards_trailing_frames(self, tmp_path):
        """Verify incomplete blocks at end of each file are discarded."""
        n = 3
        # Create two files: first has 5 frames (1 complete + 2 trailing),
        # second has 4 frames (1 complete + 1 trailing)
        paths = _make_input_files(tmp_path, [5, 4])

        out_dir = str(tmp_path / "out")

        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
            start_index=0,
        )

        # Should have 2 blocks total (1 from each file)
        total_bunches = (
            _count_bunches_for_stem(out_dir, "input_000", n)
            + _count_bunches_for_stem(out_dir, "input_001", n)
        )
        assert total_bunches == 2

    def test_continuous_vs_single_file_difference(self, tmp_path):
        """Verify continuous and single-file modes produce different results."""
        n = 3
        # Create two files with 4 frames each
        paths = _make_input_files(tmp_path, [4, 4])

        # Single-file mode
        single_dir = str(tmp_path / "single")
        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=single_dir,
            start_index=0,
        )

        # Continuous mode
        cont_dir = str(tmp_path / "continuous")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=cont_dir,
            start_index=0,
        )

        S = generate_s_matrix(n)
        first_tag = "".join(str(int(x)) for x in S[0])

        # Single: each file produces its own dominant stem
        single_f0 = _read_h5(str(Path(single_dir) / f"input_000_{first_tag}.h5"))
        single_f1 = _read_h5(str(Path(single_dir) / f"input_001_{first_tag}.h5"))

        # Continuous: first block dominant is input_000, second is input_001
        cont_f0 = _read_h5(str(Path(cont_dir) / f"input_000_{first_tag}.h5"))

        # Both single-file first block and continuous first block should be the same
        # (both use frames 0-2 from first file)
        assert single_f0.shape[0] == 1
        assert cont_f0.shape[0] == 1
        assert np.allclose(single_f0[0], cont_f0[0])

        # Second block should be different:
        # - single-file: frames 0-2 from second file (input_001)
        # - continuous: frame 3 from first file + 0-1 from second file (dominant input_001)
        cont_f1 = _read_h5(str(Path(cont_dir) / f"input_001_{first_tag}.h5"))
        assert not np.allclose(single_f1[0], cont_f1[0])

    def test_start_index_only_affects_first_file(self, tmp_path):
        """Verify start_index only applies to the first file in single-file mode."""
        n = 3
        # Create two files with 6 frames each
        paths = _make_input_files(tmp_path, [6, 6])

        out_dir = str(tmp_path / "out")

        # Start at frame 3 (should skip first block of first file)
        single_file_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
            start_index=3,
        )

        # Should have 3 blocks: 1 from first file (skipped first block) + 2 from second file
        total_bunches = (
            _count_bunches_for_stem(out_dir, "input_000", n)
            + _count_bunches_for_stem(out_dir, "input_001", n)
        )
        assert total_bunches == 3
