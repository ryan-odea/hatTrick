from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pytest

from hatTrick._helpers import (
    compute_hadamard_inverse,
    continuous_hadamard_encode,
    generate_s_matrix,
    resolve_file_list,
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
) -> Tuple[List[str], np.ndarray]:
    """Write frames split across multiple HDF5 files and return paths + ground truth.

    :param directory: Directory in which to create files (must already exist)
    :param frames_per_file: Number of frames to write in each file, in order
    :param seed: RNG seed for reproducibility
    :returns: Tuple of (file_paths, ground_truth) where file_paths is an ordered list of created file paths, and ground_truth is an ndarray of shape (sum(frames_per_file), H, W) containing all frames concatenated in global order as float32
    """
    rng = np.random.default_rng(seed)
    total = sum(frames_per_file)
    ground_truth = rng.uniform(10.0, 500.0, (total, *FRAME_SHAPE)).astype(np.float32)

    paths = []
    offset = 0
    for i, count in enumerate(frames_per_file):
        p = str(directory / f"input_{i:03d}.h5")
        _write_h5(p, ground_truth[offset : offset + count])
        paths.append(p)
        offset += count

    return paths, ground_truth


def _decode_pattern_files(
    pattern_files: List[str],
    n: int,
) -> np.ndarray:
    """Read per-pattern encoded HDF5 files and apply S^-1 to recover frames.

    :param pattern_files: One path per encoding pattern, in pattern order (pattern 0 first)
    :param n: S-matrix order
    :returns: Recovered frames in original global order with shape (n_bunches * n, H, W)
    """
    S = generate_s_matrix(n)
    Sinv = compute_hadamard_inverse(S)

    # Read encoded data for every pattern: shape (n, n_bunches, H, W).
    pattern_arrays = [_read_h5(p) for p in pattern_files]
    encoded = np.stack(pattern_arrays, axis=0)  # (n, n_bunches, H, W)

    n_patterns, n_bunches, H, W = encoded.shape
    assert n_patterns == n

    recovered = np.zeros((n_bunches, n, H, W), dtype=encoded.dtype)

    for bunch_idx in range(n_bunches):
        # E[:, px, py] is the n-vector of encoded values for one pixel.
        E = encoded[:, bunch_idx].reshape(n, -1)  # (n, H*W)
        X = Sinv @ E  # (n, H*W)
        recovered[bunch_idx] = X.reshape(n, H, W)

    # (n_bunches, n, H, W) -> (n_bunches * n, H, W) in original order.
    return recovered.reshape(n_bunches * n, H, W)


def _find_pattern_files(output_dir: str, dominant_stem: str, n: int) -> List[str]:
    """Return the per-pattern output files in pattern order.

    Files are named ``{dominant_stem}_{binary_tag}.h5`` inside *output_dir*.
    """
    S = generate_s_matrix(n)
    out = Path(output_dir)
    paths = []
    for row in S:
        tag = "".join(str(int(x)) for x in row)
        paths.append(str(out / f"{dominant_stem}_{tag}.h5"))
    return paths


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestContinuousHadamardRoundTrip:
    """Verify S^-1 @ (S @ X) == X through the full encode/decode pipeline."""

    @pytest.mark.parametrize("n", [3, 7])
    def test_single_file(self, tmp_path, n):
        """Single input file: all blocks stay within one file."""
        n_bunches = 4
        total = n * n_bunches
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, gt = _make_input_files(in_dir, [total], seed=0)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        recovered = _decode_pattern_files(
            _find_pattern_files(out_dir, "input_000", n), n
        )

        np.testing.assert_allclose(
            recovered,
            gt[: n_bunches * n],
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Single-file round-trip failed for n={n}",
        )

    @pytest.mark.parametrize("n", [3, 7])
    def test_blocks_split_evenly_across_files(self, tmp_path, n):
        """One complete block per file: no block spans a boundary."""
        n_bunches = 4
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # n frames per file -> one block per file, n files
        paths, gt = _make_input_files(in_dir, [n] * n_bunches, seed=1)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        # Each file is its own dominant, so we need to collect across all stems
        # and decode per-stem, then concatenate.
        all_recovered = []
        for i in range(n_bunches):
            stem = f"input_{i:03d}"
            pfiles = _find_pattern_files(out_dir, stem, n)
            rec = _decode_pattern_files(pfiles, n)
            all_recovered.append(rec)
        recovered = np.concatenate(all_recovered, axis=0)

        np.testing.assert_allclose(
            recovered,
            gt,
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Even-split round-trip failed for n={n}",
        )

    @pytest.mark.parametrize("n", [3, 7])
    def test_blocks_span_file_boundaries(self, tmp_path, n):
        """File sizes chosen so every block crosses at least one file boundary."""
        n_bunches = 3
        total = n * n_bunches
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # Split so cuts fall mid-block: e.g. for n=3 total=9 -> [2, 4, 3]
        cut = n - 1
        sizes = [cut, total - 2 * cut, cut]
        sizes = [max(1, s) for s in sizes]
        paths, gt = _make_input_files(in_dir, sizes, seed=2)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        # Find all pattern files in the output directory and group by dominant stem
        recovered = _collect_and_decode(out_dir, n)
        n_recovered = recovered.shape[0]

        np.testing.assert_allclose(
            recovered,
            gt[:n_recovered],
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Cross-boundary round-trip failed for n={n}",
        )

    @pytest.mark.parametrize("n", [3, 7])
    def test_one_frame_per_file(self, tmp_path, n):
        """Extreme case: one frame per file, every block spans n file boundaries."""
        n_bunches = 3
        total = n * n_bunches
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, gt = _make_input_files(in_dir, [1] * total, seed=3)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        recovered = _collect_and_decode(out_dir, n)

        np.testing.assert_allclose(
            recovered,
            gt,
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"One-frame-per-file round-trip failed for n={n}",
        )

    @pytest.mark.parametrize("n", [3, 7])
    def test_start_index_skips_frames(self, tmp_path, n):
        """Frames before start_index are excluded from all encoded bunches."""
        skip = n  # skip exactly one block's worth of frames
        n_bunches = 2
        total = n * (n_bunches + 1)  # extra block to skip at the front
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, gt = _make_input_files(in_dir, [total], seed=4)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
            start_index=skip,
        )

        recovered = _decode_pattern_files(
            _find_pattern_files(out_dir, "input_000", n), n
        )

        # Ground truth begins at frame `skip`.
        np.testing.assert_allclose(
            recovered,
            gt[skip : skip + n_bunches * n],
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"start_index round-trip failed for n={n}",
        )

    @pytest.mark.parametrize("n", [3, 7])
    def test_start_index_spans_file_boundary(self, tmp_path, n):
        """start_index that falls inside the second file is handled correctly."""
        n_bunches = 2
        first_file_frames = n + 1  # start_index falls inside file 1
        second_file_frames = n * n_bunches
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, gt = _make_input_files(in_dir, [first_file_frames, second_file_frames], seed=5)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
            start_index=first_file_frames,
        )

        recovered = _decode_pattern_files(
            _find_pattern_files(out_dir, "input_001", n), n
        )

        np.testing.assert_allclose(
            recovered,
            gt[first_file_frames : first_file_frames + n_bunches * n],
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Cross-file start_index round-trip failed for n={n}",
        )

    def test_output_files_named_by_dominant_stem(self, tmp_path):
        """Output filenames contain the dominant input stem and binary S-matrix row."""
        n = 3
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, _ = _make_input_files(in_dir, [n * 2], seed=6)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        S = generate_s_matrix(n)
        out = Path(out_dir)
        for row in S:
            tag = "".join(str(int(x)) for x in row)
            expected = out / f"input_000_{tag}.h5"
            assert expected.is_file(), f"Expected output file missing: {expected}"

    def test_encoded_shape(self, tmp_path):
        """Each pattern file contains (n_bunches, H, W) data."""
        n = 3
        n_bunches = 4
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, _ = _make_input_files(in_dir, [n * n_bunches], seed=7)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        for p in _find_pattern_files(out_dir, "input_000", n):
            data = _read_h5(p)
            assert data.shape == (
                n_bunches,
                *FRAME_SHAPE,
            ), f"Pattern file {p} has unexpected shape {data.shape}"

    def test_trailing_incomplete_block_discarded(self, tmp_path):
        """Frames that do not fill a complete block are silently discarded."""
        n = 3
        n_bunches = 2
        leftover = n - 1
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, _ = _make_input_files(in_dir, [n * n_bunches + leftover], seed=8)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=paths,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        for p in _find_pattern_files(out_dir, "input_000", n):
            data = _read_h5(p)
            assert (
                data.shape[0] == n_bunches
            ), f"Expected {n_bunches} bunches but got {data.shape[0]} in {p}"

    def test_resolve_file_list_glob(self, tmp_path):
        """resolve_file_list with a glob feeds correctly into encode."""
        n = 3
        n_bunches = 2
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, gt = _make_input_files(in_dir, [n * n_bunches], seed=9)

        # Use a glob instead of explicit paths.
        glob_pattern = str(in_dir / "input_*.h5")
        resolved = resolve_file_list(glob_pattern)

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=resolved,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        recovered = _decode_pattern_files(
            _find_pattern_files(out_dir, "input_000", n), n
        )
        np.testing.assert_allclose(recovered, gt, rtol=1e-4, atol=1e-3)

    def test_resolve_file_list_manifest(self, tmp_path):
        """resolve_file_list with a list-file feeds correctly into encode."""
        n = 3
        n_bunches = 2
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        paths, gt = _make_input_files(in_dir, [n] * n_bunches, seed=10)

        manifest = tmp_path / "files.txt"
        manifest.write_text("\n".join(paths) + "\n")

        resolved = resolve_file_list(str(manifest))

        out_dir = str(tmp_path / "out")
        continuous_hadamard_encode(
            file_paths=resolved,
            n_merged_frames=n,
            data_location=DATA_LOCATION,
            data_name=DATA_NAME,
            output_dir=out_dir,
        )

        # Each file is its own dominant stem
        all_recovered = []
        for i in range(n_bunches):
            stem = f"input_{i:03d}"
            pfiles = _find_pattern_files(out_dir, stem, n)
            rec = _decode_pattern_files(pfiles, n)
            all_recovered.append(rec)
        recovered = np.concatenate(all_recovered, axis=0)
        np.testing.assert_allclose(recovered, gt, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# Helper to collect all dominant-stem groups and decode in order
# ---------------------------------------------------------------------------


def _collect_and_decode(output_dir: str, n: int) -> np.ndarray:
    """Find all output groups in *output_dir*, decode each, and concatenate.

    Groups are identified by finding all files matching ``*_{tag}.h5`` for the
    first S-matrix row tag, then extracting the dominant stem prefix.  Groups
    are decoded in natsorted order by stem.
    """
    from natsort import natsorted

    S = generate_s_matrix(n)
    first_tag = "".join(str(int(x)) for x in S[0])
    suffix = f"_{first_tag}.h5"

    out = Path(output_dir)
    # Find all files for the first pattern
    first_pattern_files = sorted(out.glob(f"*{suffix}"))

    stems = []
    for fp in first_pattern_files:
        # Strip the _{tag}.h5 suffix to get the dominant stem
        stem = fp.name[: -len(suffix)]
        stems.append(stem)

    stems = natsorted(stems)

    all_recovered = []
    for stem in stems:
        pfiles = _find_pattern_files(output_dir, stem, n)
        rec = _decode_pattern_files(pfiles, n)
        all_recovered.append(rec)

    return np.concatenate(all_recovered, axis=0)
