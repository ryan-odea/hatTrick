from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest

from hatTrick._helpers import _frame_iterator, continuous_hadamard_encode, resolve_file_list

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_LOCATION = "entry/data"
DATA_NAME = "data"
FRAME_SHAPE = (4, 4)


def _write_h5(path: str, frames: np.ndarray) -> None:
    """Write *frames* (shape: n, H, W) to an HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(DATA_LOCATION)
        grp.create_dataset(DATA_NAME, data=frames)


def _make_files(tmp_path: Path, frame_groups: List[np.ndarray]) -> List[str]:
    """Create one HDF5 file per element of *frame_groups* and return their paths."""
    paths = []
    for i, frames in enumerate(frame_groups):
        p = str(tmp_path / f"run_{i:03d}.h5")
        _write_h5(p, frames)
        paths.append(p)
    return paths


def _read_output(path: str) -> np.ndarray:
    """Read the merged dataset back from *path*."""
    with h5py.File(path, "r") as f:
        return f[f"{DATA_LOCATION}/{DATA_NAME}"][:]


def _find_pattern_files(output_dir: str, dominant_stem: str, n: int) -> List[str]:
    """Return the per-pattern output files in pattern order."""
    from hatTrick._helpers import generate_s_matrix

    S = generate_s_matrix(n)
    out = Path(output_dir)
    paths = []
    for row in S:
        tag = "".join(str(int(x)) for x in row)
        paths.append(str(out / f"{dominant_stem}_{tag}.h5"))
    return paths


# ---------------------------------------------------------------------------
# resolve_file_list
# ---------------------------------------------------------------------------


class TestResolveFileList:
    def test_glob_expansion(self, tmp_path):
        """Glob pattern resolves to sorted file list."""
        for i in range(3):
            (tmp_path / f"run_{i}.h5").write_text("")
        result = resolve_file_list(str(tmp_path / "run_*.h5"))
        assert len(result) == 3
        assert result == sorted(result)

    def test_list_file(self, tmp_path):
        """Plain-text list file is parsed correctly."""
        h5_paths = [str(tmp_path / f"file_{i}.h5") for i in range(3)]
        list_file = tmp_path / "files.txt"
        list_file.write_text("# comment\n" + "\n".join(h5_paths) + "\n")
        result = resolve_file_list(str(list_file))
        assert result == h5_paths

    def test_list_file_ignores_blank_and_comments(self, tmp_path):
        """Blank lines and #-comments are stripped from list files."""
        list_file = tmp_path / "files.txt"
        list_file.write_text("\n# header\nfile_a.h5\n\n# another comment\nfile_b.h5\n")
        result = resolve_file_list(str(list_file))
        assert result == ["file_a.h5", "file_b.h5"]

    def test_empty_glob_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No files matched"):
            resolve_file_list(str(tmp_path / "nonexistent_*.h5"))

    def test_empty_list_file_raises(self, tmp_path):
        list_file = tmp_path / "empty.txt"
        list_file.write_text("# only comments\n\n")
        with pytest.raises(ValueError, match="contains no file paths"):
            resolve_file_list(str(list_file))


# ---------------------------------------------------------------------------
# _frame_iterator
# ---------------------------------------------------------------------------


class TestFrameIterator:
    def test_yields_all_frames_single_file(self, tmp_path):
        frames = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        paths = _make_files(tmp_path, [frames])
        result = [f for f, _, _ in _frame_iterator(paths, DATA_LOCATION, DATA_NAME)]
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], frames[0])

    def test_yields_frames_across_files(self, tmp_path):
        f1 = np.zeros((3, 2, 2), dtype=np.float32)
        f2 = np.ones((3, 2, 2), dtype=np.float32)
        paths = _make_files(tmp_path, [f1, f2])
        result = [f for f, _, _ in _frame_iterator(paths, DATA_LOCATION, DATA_NAME)]
        assert len(result) == 6
        # First 3 frames are zeros, next 3 are ones.
        np.testing.assert_array_equal(result[2], np.zeros((2, 2)))
        np.testing.assert_array_equal(result[3], np.ones((2, 2)))

    def test_start_index_skips_frames(self, tmp_path):
        frames = np.arange(12, dtype=np.float32).reshape(4, 3)
        paths = _make_files(tmp_path, [frames])
        result = [f for f, _, _ in _frame_iterator(paths, DATA_LOCATION, DATA_NAME, start_index=2)]
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], frames[2])

    def test_start_index_spans_files(self, tmp_path):
        """start_index beyond the first file skips into the second."""
        f1 = np.zeros((4, *FRAME_SHAPE), dtype=np.float32)
        f2 = np.ones((4, *FRAME_SHAPE), dtype=np.float32)
        paths = _make_files(tmp_path, [f1, f2])
        result = [f for f, _, _ in _frame_iterator(paths, DATA_LOCATION, DATA_NAME, start_index=4)]
        assert len(result) == 4
        np.testing.assert_array_equal(result[0], np.ones(FRAME_SHAPE))

    def test_yields_file_path(self, tmp_path):
        """_frame_iterator yields the source file path for each frame."""
        f1 = np.zeros((2, *FRAME_SHAPE), dtype=np.float32)
        f2 = np.ones((2, *FRAME_SHAPE), dtype=np.float32)
        paths = _make_files(tmp_path, [f1, f2])
        result = [(fp, f) for f, _, fp in _frame_iterator(paths, DATA_LOCATION, DATA_NAME)]
        assert result[0][0] == paths[0]
        assert result[1][0] == paths[0]
        assert result[2][0] == paths[1]
        assert result[3][0] == paths[1]


# ---------------------------------------------------------------------------
# continuous_rolling_merge -- correctness
# ---------------------------------------------------------------------------


class TestContinuousHadamardEncode:
    def test_single_file_single_block(self, tmp_path):
        """Three frames in one file -> one merged bunch per pattern."""
        frames = np.array(
            [
                np.full(FRAME_SHAPE, 1.0),
                np.full(FRAME_SHAPE, 2.0),
                np.full(FRAME_SHAPE, 3.0),
            ],
            dtype=np.float32,
        )
        paths = _make_files(tmp_path, [frames])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )
        for p in _find_pattern_files(out_dir, "run_000", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)

    def test_block_spans_file_boundary(self, tmp_path):
        """A block of 3 frames that spans two files is encoded correctly."""
        f1 = np.array(
            [np.full(FRAME_SHAPE, 1.0), np.full(FRAME_SHAPE, 2.0)],
            dtype=np.float32,
        )
        f2 = np.array(
            [np.full(FRAME_SHAPE, 3.0), np.full(FRAME_SHAPE, 4.0)],
            dtype=np.float32,
        )
        paths = _make_files(tmp_path, [f1, f2])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )
        # Block of 3 from frames [1,2,3]: 2 from run_000, 1 from run_001 -> dominant is run_000
        for p in _find_pattern_files(out_dir, "run_000", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)

    def test_multiple_blocks_across_files(self, tmp_path):
        """Two complete blocks spread across three files."""
        frame_data = [float(i) for i in range(6)]
        all_frames = np.array([np.full(FRAME_SHAPE, v) for v in frame_data], dtype=np.float32)
        paths = _make_files(tmp_path, [all_frames[0:2], all_frames[2:4], all_frames[4:6]])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )
        # Block 1: frames 0,1,2 -> run_000(2), run_001(1) -> dominant run_000
        # Block 2: frames 3,4,5 -> run_001(1), run_002(2) -> dominant run_002
        for p in _find_pattern_files(out_dir, "run_000", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)
        for p in _find_pattern_files(out_dir, "run_002", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)

    def test_start_index(self, tmp_path):
        """Frames before start_index are skipped."""
        frames = np.array([np.full(FRAME_SHAPE, float(i)) for i in range(6)], dtype=np.float32)
        paths = _make_files(tmp_path, [frames])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            start_index=3,
            output_dir=out_dir,
        )
        for p in _find_pattern_files(out_dir, "run_000", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)

    def test_incomplete_trailing_block_discarded(self, tmp_path):
        """A trailing block with fewer than n_merged_frames frames is discarded."""
        frames = np.array([np.full(FRAME_SHAPE, float(i)) for i in range(5)], dtype=np.float32)
        paths = _make_files(tmp_path, [frames])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )
        for p in _find_pattern_files(out_dir, "run_000", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)

    def test_no_frames_after_start_raises(self, tmp_path):
        """start_index beyond all frames raises ValueError."""
        frames = np.zeros((3, *FRAME_SHAPE), dtype=np.float32)
        paths = _make_files(tmp_path, [frames])
        out_dir = str(tmp_path / "out")
        with pytest.raises(ValueError, match="No frames available"):
            continuous_hadamard_encode(
                file_paths=paths,
                n_merged_frames=3,
                data_location=DATA_LOCATION,
                data_name=DATA_NAME,
                start_index=99,
                output_dir=out_dir,
            )

    def test_start_index_spans_file_boundary(self, tmp_path):
        """start_index that falls inside the second file is handled correctly."""
        f1 = np.array([np.full(FRAME_SHAPE, float(i)) for i in range(4)], dtype=np.float32)
        f2 = np.array([np.full(FRAME_SHAPE, float(i + 4)) for i in range(4)], dtype=np.float32)
        paths = _make_files(tmp_path, [f1, f2])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            start_index=5,
            output_dir=out_dir,
        )
        for p in _find_pattern_files(out_dir, "run_001", 3):
            result = _read_output(p)
            assert result.shape == (1, *FRAME_SHAPE)

    def test_output_naming_convention(self, tmp_path):
        """Output files are named with dominant stem and pattern-specific tags."""
        frames = np.zeros((3, *FRAME_SHAPE), dtype=np.float32)
        paths = _make_files(tmp_path, [frames])
        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=3,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )
        from hatTrick._helpers import generate_s_matrix

        S = generate_s_matrix(3)
        out = Path(out_dir)
        for row in S:
            tag = "".join(str(int(x)) for x in row)
            pattern_file = out / f"run_000_{tag}.h5"
            assert pattern_file.is_file()
