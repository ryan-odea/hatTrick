from pathlib import Path

import numpy as np
import pytest

from hatTrick._helpers import generate_s_matrix
from hatTrick import decode_hadamard_files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_crystfel_simple(
    path: str, h: np.ndarray, k: np.ndarray, l: np.ndarray, I: np.ndarray, sigI: np.ndarray
) -> None:
    """Write a minimal crystfel_simple reflection file (5-column)."""
    with open(path, "w") as f:
        f.write("CrystFEL reflection list version 2.0\n")
        f.write("Symmetry: 1\n")
        f.write("   h    k    l          I   sigma(I)\n")
        for hi, ki, li, Ii, si in zip(h, k, l, I, sigI):
            f.write(f"{hi:4d} {ki:4d} {li:4d} {Ii:12.4f} {si:12.4f}\n")
        f.write("End of reflections\n")


def _read_hkl(path: str, skip: int = 3) -> np.ndarray:
    """Read a decoded .hkl file, skipping *skip* header lines."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i < skip or not line or line.startswith("#") or line == "End of reflections":
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)


def _make_encoded_files(tmp_path: Path, n: int, n_bunches: int, seed: int = 0):
    """Produce synthetic encoded reflection files and return their paths.

    Generates ground-truth frame intensities, applies the S-matrix, writes
    one file per (pattern, bunch) pair, and returns both the file path lists
    and the ground-truth data so tests can verify round-trip correctness.

    :param tmp_path: Temporary directory for creating test files
    :param n: S-matrix order (number of encoding patterns)
    :param n_bunches: Number of bunches to generate
    :param seed: RNG seed for reproducibility
    :returns: Tuple of (encoded_files, ground_truth_I, h, k, l, sigI_per_frame) where encoded_files is a list of list of str (outer index: pattern; inner index: bunch), ground_truth_I is ndarray of shape (n, n_bunches, n_reflections) containing ground-truth per-frame intensities, h/k/l are Miller indices, and sigI_per_frame is ndarray of shape (n, n_bunches, n_reflections) containing sigma values
    """
    rng = np.random.default_rng(seed)
    n_reflections = 20

    h = rng.integers(-5, 6, n_reflections)
    k = rng.integers(-5, 6, n_reflections)
    l = rng.integers(0, 6, n_reflections)

    S = generate_s_matrix(n)

    # Ground truth: n frames, n_bunches, n_reflections
    ground_truth_I = rng.uniform(10.0, 1000.0, (n, n_bunches, n_reflections))
    sigI_per_frame = ground_truth_I * 0.05

    # Encode: for each bunch, E = S @ ground_truth (per-reflection)
    encoded_I = np.einsum("pi,ibr->pbr", S.astype(float), ground_truth_I)
    encoded_sigI = np.sqrt(np.einsum("pi,ibr->pbr", S.astype(float) ** 2, sigI_per_frame**2))

    encoded_files = []
    for pattern_idx in range(n):
        pattern_files = []
        for bunch_idx in range(n_bunches):
            fpath = str(tmp_path / f"enc_p{pattern_idx}_b{bunch_idx}.hkl")
            _write_crystfel_simple(
                fpath,
                h,
                k,
                l,
                encoded_I[pattern_idx, bunch_idx],
                encoded_sigI[pattern_idx, bunch_idx],
            )
            pattern_files.append(fpath)
        encoded_files.append(pattern_files)

    return encoded_files, ground_truth_I, h, k, l, sigI_per_frame


class TestDecodeIntensityRoundTrip:
    def test_single_bunch_round_trip(self, tmp_path):
        """Decoded intensities recover ground-truth values for n=3, 1 bunch."""
        n = 3
        encoded_files, gt_I, h, k, l, _ = _make_encoded_files(tmp_path, n, n_bunches=1)

        result = decode_hadamard_files(
            encoded_files=encoded_files,
            n_merged_frames=n,
            file_format="crystfel_simple",
        )

        assert len(result) == 1
        df = result[0]
        for frame_idx in range(n):
            decoded_I = df[f"I_frame{frame_idx}"].values
            # Ground truth is ordered by reflection; align via hkl merge.
            np.testing.assert_allclose(decoded_I, gt_I[frame_idx, 0], rtol=1e-4)

    def test_multi_bunch_round_trip(self, tmp_path):
        """Decoded intensities recover ground truth for n=3, 3 bunches."""
        n = 3
        n_bunches = 3
        encoded_files, gt_I, h, k, l, _ = _make_encoded_files(tmp_path, n, n_bunches)

        result = decode_hadamard_files(
            encoded_files=encoded_files,
            n_merged_frames=n,
            file_format="crystfel_simple",
        )

        assert len(result) == n_bunches
        for bunch_idx, df in enumerate(result):
            for frame_idx in range(n):
                np.testing.assert_allclose(
                    df[f"I_frame{frame_idx}"].values,
                    gt_I[frame_idx, bunch_idx],
                    rtol=1e-4,
                )

    def test_output_files_written(self, tmp_path):
        """Output .hkl files are created for each bunch and frame."""
        n = 3
        enc_dir = tmp_path / "enc"
        enc_dir.mkdir()
        encoded_files, _, _, _, _, _ = _make_encoded_files(enc_dir, n, n_bunches=2)
        out_dir = str(tmp_path / "decoded")

        decode_hadamard_files(
            encoded_files=encoded_files,
            n_merged_frames=n,
            output_dir=out_dir,
            output_prefix="test",
            file_format="crystfel_simple",
        )

        for bunch_idx in range(2):
            for frame_idx in range(n):
                expected = Path(out_dir) / f"test_frame{bunch_idx * n + frame_idx}.hkl"
                assert expected.is_file(), f"Missing: {expected}"

    def test_output_files_not_written_without_output_dir(self, tmp_path):
        """No files written when output_dir is None."""
        n = 3
        encoded_files, _, _, _, _, _ = _make_encoded_files(tmp_path, n, n_bunches=1)
        decode_hadamard_files(
            encoded_files=encoded_files,
            n_merged_frames=n,
            file_format="crystfel_simple",
        )
        hkl_files = list(tmp_path.glob("*.hkl"))
        # Only the input files exist; no decoded outputs.
        assert all("enc_" in f.name for f in hkl_files)


# ---------------------------------------------------------------------------
class TestResolveFileListInDecode:
    def test_mismatched_pattern_counts_raises(self, tmp_path):
        """Unequal file counts across patterns raise ValueError."""
        n = 3
        encoded_files, _, _, _, _, _ = _make_encoded_files(tmp_path, n, n_bunches=2)
        # Drop one file from the last pattern.
        encoded_files[-1] = encoded_files[-1][:1]

        with pytest.raises(ValueError, match="same number of files"):
            decode_hadamard_files(
                encoded_files=encoded_files,
                n_merged_frames=n,
                file_format="crystfel_simple",
            )
