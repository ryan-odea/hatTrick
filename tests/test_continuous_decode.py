import glob as globmod

import numpy as np
import pytest

from hatTrick._helpers import generate_s_matrix
from hatTrick.decode.HATRX import decode_hadamard_files


@pytest.fixture
def s_matrix():
    return generate_s_matrix(3)


@pytest.fixture
def frame_values():
    """9 original frames (3 bunches of 3), each with a distinct I value."""
    return [10.0 * (i + 1) for i in range(9)]


def _create_encoded_files_continuous(tmp_dir, S, frame_values, n, offset):
    """
    Simulate continuous Hadamard encoding with a constant S-row rotation.

    All bunches use S rows rotated by `offset`:
      measurement i within any bunch uses S[(i + offset) % n]

    Bunches are non-overlapping: bunch b covers frames b*n .. b*n+n-1.
    """
    n_bunches = len(frame_values) // n
    sigma = 1.0
    reflections = [(1, 0, 0), (0, 1, 0)]

    # encoded_files[pattern_idx][bunch_idx] = file path
    encoded_files = [[] for _ in range(n)]

    for bunch_idx in range(n_bunches):
        base_frame = bunch_idx * n
        for meas_idx in range(n):
            s_row = S[(meas_idx + offset) % n]
            encoded_I = sum(
                s_row[j] * frame_values[base_frame + j] for j in range(n)
            )
            encoded_sigma = sum(s_row[j] * sigma for j in range(n))

            file_path = tmp_dir / f"bunch{bunch_idx}_meas{meas_idx}.hkl"
            with open(file_path, "w") as f:
                for h, k, l in reflections:
                    f.write(f"{h} {k} {l} {encoded_I:.4f} {encoded_sigma:.4f}\n")

            encoded_files[meas_idx].append(str(file_path))

    return encoded_files


def _create_encoded_files_standard(tmp_dir, S, frame_values, n):
    """Standard (non-continuous) encoding: S rows in natural order."""
    n_bunches = len(frame_values) // n
    sigma = 1.0
    reflections = [(1, 0, 0), (0, 1, 0)]

    encoded_files = [[] for _ in range(n)]

    for bunch_idx in range(n_bunches):
        base_frame = bunch_idx * n
        for pattern_idx in range(n):
            s_row = S[pattern_idx]
            encoded_I = sum(
                s_row[j] * frame_values[base_frame + j] for j in range(n)
            )
            encoded_sigma = sum(s_row[j] * sigma for j in range(n))

            file_path = tmp_dir / f"enc_b{bunch_idx}_p{pattern_idx}.hkl"
            with open(file_path, "w") as f:
                for h, k, l in reflections:
                    f.write(f"{h} {k} {l} {encoded_I:.4f} {encoded_sigma:.4f}\n")

            encoded_files[pattern_idx].append(str(file_path))

    return encoded_files


def test_non_continuous_naming(s_matrix, frame_values, tmp_path):
    """Without --continuous, output files should be named frame0, frame1, etc."""
    n = 3
    encoded_files = _create_encoded_files_standard(tmp_path, s_matrix, frame_values, n)

    output_dir = tmp_path / "decoded"
    decode_hadamard_files(
        encoded_files=encoded_files,
        n_merged_frames=n,
        output_dir=str(output_dir),
        output_prefix="test",
        file_format="crystfel_simple",
    )

    n_bunches = len(frame_values) // n
    for i in range(n_bunches * n):
        assert (output_dir / f"test_frame{i}.hkl").exists(), f"Missing test_frame{i}.hkl"

    assert len(globmod.glob(str(output_dir / "*bunch*"))) == 0


def test_non_continuous_data_correctness(s_matrix, frame_values, tmp_path):
    """Standard decode should recover original frame I values."""
    n = 3
    encoded_files = _create_encoded_files_standard(tmp_path, s_matrix, frame_values, n)

    output_dir = tmp_path / "decoded"
    results = decode_hadamard_files(
        encoded_files=encoded_files,
        n_merged_frames=n,
        output_dir=str(output_dir),
        output_prefix="test",
        file_format="crystfel_simple",
    )

    for bunch_idx, df in enumerate(results):
        for j in range(n):
            decoded_I = df[f"I_frame{j}"].iloc[0]
            expected_I = frame_values[bunch_idx * n + j]
            np.testing.assert_allclose(
                decoded_I, expected_I, atol=1e-6,
                err_msg=f"Bunch {bunch_idx} frame {j}: expected {expected_I}, got {decoded_I}"
            )


def test_continuous_offset1_frame_ordering(s_matrix, frame_values, tmp_path):
    """
    With continuous offset=1:
    - Create 3 bunches of continuous data (9 frames, offset=1)
    - Drop first and last (boundary effects), keep middle bunch
    - Decode with --continuous 1
    - Verify output frames contain correct I values in order
    """
    n = 3
    offset = 1

    all_encoded = _create_encoded_files_continuous(
        tmp_path, s_matrix, frame_values, n, offset
    )

    # Drop first and last bunches, keep bunch 1 (frames F3, F4, F5)
    trimmed_files = [pattern_files[1:2] for pattern_files in all_encoded]
    assert len(trimmed_files[0]) == 1

    output_dir = tmp_path / "decoded_continuous"
    results = decode_hadamard_files(
        encoded_files=trimmed_files,
        n_merged_frames=n,
        output_dir=str(output_dir),
        output_prefix="test",
        file_format="crystfel_simple",
        continuous_offset=offset,
    )

    # After correction, frame0=F3(40), frame1=F4(50), frame2=F5(60)
    expected = {
        0: frame_values[3],  # F3 = 40
        1: frame_values[4],  # F4 = 50
        2: frame_values[5],  # F5 = 60
    }

    for frame_num, expected_I in expected.items():
        frame_file = output_dir / f"test_frame{frame_num}.hkl"
        assert frame_file.exists(), f"Missing test_frame{frame_num}.hkl"

        with open(frame_file) as f:
            lines = f.readlines()

        for line in lines:
            stripped = line.strip()
            if (stripped and not stripped.startswith("#")
                    and not stripped.startswith("CrystFEL")
                    and not stripped.startswith("Symmetry")
                    and not stripped.startswith("End")
                    and "h" not in stripped):
                I_val = float(stripped.split()[3])
                np.testing.assert_allclose(
                    I_val, expected_I, atol=1e-4,
                    err_msg=f"frame{frame_num}: expected I={expected_I}, got I={I_val}"
                )
                break


def test_continuous_multiple_bunches(s_matrix, frame_values, tmp_path):
    """
    With continuous offset=1 and multiple kept bunches, all frames should
    be correctly ordered sequentially.
    """
    n = 3
    offset = 1
    # Use 12 frames (4 bunches), drop first and last, keep bunches 1 and 2
    fvals = [10.0 * (i + 1) for i in range(12)]

    all_encoded = _create_encoded_files_continuous(
        tmp_path, s_matrix, fvals, n, offset
    )

    trimmed = [files[1:-1] for files in all_encoded]
    assert len(trimmed[0]) == 2  # bunches 1 and 2

    output_dir = tmp_path / "decoded"
    results = decode_hadamard_files(
        encoded_files=trimmed,
        n_merged_frames=n,
        output_dir=str(output_dir),
        output_prefix="test",
        file_format="crystfel_simple",
        continuous_offset=offset,
    )

    # Bunch 1 covers F3,F4,F5; Bunch 2 covers F6,F7,F8
    # Output: frame0=F3(40), frame1=F4(50), ..., frame5=F8(90)
    for i in range(6):
        frame_file = output_dir / f"test_frame{i}.hkl"
        assert frame_file.exists(), f"Missing test_frame{i}.hkl"

        expected_I = fvals[3 + i]  # starts at F3

        with open(frame_file) as f:
            lines = f.readlines()

        for line in lines:
            stripped = line.strip()
            if (stripped and not stripped.startswith("#")
                    and not stripped.startswith("CrystFEL")
                    and not stripped.startswith("Symmetry")
                    and not stripped.startswith("End")
                    and "h" not in stripped):
                I_val = float(stripped.split()[3])
                np.testing.assert_allclose(
                    I_val, expected_I, atol=1e-4,
                    err_msg=f"frame{i}: expected I={expected_I}, got {I_val}"
                )
                break


def test_continuous_no_bunch_naming(s_matrix, frame_values, tmp_path):
    """Continuous decode should use frame-only naming."""
    n = 3
    all_encoded = _create_encoded_files_continuous(
        tmp_path, s_matrix, frame_values, n, 1
    )
    trimmed = [files[1:2] for files in all_encoded]

    output_dir = tmp_path / "decoded"
    decode_hadamard_files(
        encoded_files=trimmed,
        n_merged_frames=n,
        output_dir=str(output_dir),
        output_prefix="out",
        file_format="crystfel_simple",
        continuous_offset=1,
    )

    assert len(globmod.glob(str(output_dir / "*bunch*"))) == 0
    assert len(globmod.glob(str(output_dir / "out_frame*.hkl"))) > 0


def test_continuous_vs_standard_same_data_different_order(s_matrix, tmp_path):
    """
    Verify that without correction, continuous decode gives wrong frame order,
    and with correction it gives the right order (matching standard decode).
    """
    n = 3
    fvals = [10.0, 20.0, 30.0]  # single bunch

    # Standard encoding
    std_dir = tmp_path / "std"
    std_dir.mkdir()
    std_files = _create_encoded_files_standard(std_dir, s_matrix, fvals, n)

    std_out = tmp_path / "std_out"
    std_results = decode_hadamard_files(
        encoded_files=std_files,
        n_merged_frames=n,
        output_dir=str(std_out),
        output_prefix="std",
        file_format="crystfel_simple",
    )

    # Continuous encoding with offset=1
    cont_dir = tmp_path / "cont"
    cont_dir.mkdir()
    cont_files = _create_encoded_files_continuous(cont_dir, s_matrix, fvals, n, offset=1)

    cont_out = tmp_path / "cont_out"
    cont_results = decode_hadamard_files(
        encoded_files=cont_files,
        n_merged_frames=n,
        output_dir=str(cont_out),
        output_prefix="cont",
        file_format="crystfel_simple",
        continuous_offset=1,
    )

    # Both should recover the same frame values in the same file names
    for i in range(n):
        std_file = std_out / f"std_frame{i}.hkl"
        cont_file = cont_out / f"cont_frame{i}.hkl"

        with open(std_file) as f:
            std_lines = f.readlines()
        with open(cont_file) as f:
            cont_lines = f.readlines()

        # Extract I values and compare
        def get_I(lines):
            for line in lines:
                s = line.strip()
                if (s and not s.startswith("#") and not s.startswith("CrystFEL")
                        and not s.startswith("Symmetry") and not s.startswith("End")
                        and "h" not in s):
                    return float(s.split()[3])
            return None

        np.testing.assert_allclose(
            get_I(cont_lines), get_I(std_lines), atol=1e-4,
            err_msg=f"frame{i}: continuous decode doesn't match standard decode"
        )
