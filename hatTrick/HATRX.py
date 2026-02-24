from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from ._helpers import compute_hadamard_inverse, generate_s_matrix


def _intensities_to_amplitudes(
    df: pd.DataFrame,
    pattern_idx: int,
    file_format: str,
) -> pd.DataFrame:
    """Convert intensity columns to structure-factor amplitudes in place.

    The conversion is::

        F      = sqrt(|I|)
        sigF   = sigI / (2 * F)     (standard error propagation)

    A small floor of 1e-6 is applied to F before computing sigF to avoid
    division by zero when I is near zero.  Reflections with I < 0 produce a
    real-valued F via ``sqrt(|I|)``; callers may wish to flag or discard these.

    This must be called *per pattern* **before** the Hadamard inversion because
    the transform is linear over intensities; applying it to amplitudes after
    inversion would be physically incorrect.

    :param df: DataFrame for a single encoding pattern; must contain intensity and sigma columns appropriate to *file_format*
    :param pattern_idx: Index of the encoding pattern; used to locate the correct column names
    :param file_format: One of ``"crystfel"``, ``"crystfel_simple"``, or ``"ccp4"``
    :returns: The same DataFrame with ``F_{pattern_idx}`` and ``sigF_{pattern_idx}`` columns added
    """
    I_col = f"I_{pattern_idx}"
    sig_col = (
        f"SIGMA_{pattern_idx}"
        if file_format in ("crystfel", "crystfel_simple")
        else f"sigI_{pattern_idx}"
    )

    I = df[I_col].to_numpy(dtype=float)
    sigI = df[sig_col].to_numpy(dtype=float)

    F = np.sqrt(np.abs(I))
    F_safe = np.where(F < 1e-6, 1e-6, F)
    sigF = sigI / (2.0 * F_safe)

    df = df.copy()
    df[f"F_{pattern_idx}"] = F
    df[f"sigF_{pattern_idx}"] = sigF
    return df


def decode_hadamard_files(
    encoded_files: List[List[str]],
    n_merged_frames: int = 3,
    output_dir: Optional[str] = None,
    output_prefix: str = "decoded",
    S_matrix: Optional[np.ndarray] = None,
    file_format: str = "auto",
    decode_amplitudes: bool = False,
) -> List[pd.DataFrame]:
    """Decode Hadamard-encoded crystallographic reflection files.

    Each element of *encoded_files* is an ordered list of file paths for one
    encoding pattern.  Paths may be pre-resolved or produced by
    :func:`resolve_file_list`.  All pattern lists must have the same length
    (one file per bunch).

    When *decode_amplitudes* is ``True``, structure-factor amplitudes are
    derived from intensities via ``F = sqrt(|I|)`` **before** the Hadamard
    inversion.  This is the physically correct order because the inversion is
    a linear operation on intensities; applying it to amplitudes post-inversion
    would not be equivalent.  Amplitude-decoded frames are written to separate
    output files with a ``_F`` suffix.

    :param encoded_files: ``n_merged_frames`` lists, each containing the paths for one encoding pattern sorted by bunch index
    :param n_merged_frames: Number of encoded patterns (order of the Hadamard S-matrix)
    :param output_dir: Directory in which to write decoded output files; created if absent; if ``None``, no files are written
    :param output_prefix: Stem prefix for output filenames
    :param S_matrix: Pre-computed S-matrix; generated automatically when ``None``
    :param file_format: ``"auto"`` (detect from first file), ``"crystfel"``, ``"crystfel_simple"``, or ``"ccp4"``
    :param decode_amplitudes: When ``True``, also decode structure-factor amplitudes converted from intensities prior to inversion
    :returns: One DataFrame per bunch containing all decoded frames
    """
    if len(encoded_files) != n_merged_frames:
        raise ValueError(f"Expected {n_merged_frames} file lists, got {len(encoded_files)}")

    n_bunches = len(encoded_files[0])
    for i, file_list in enumerate(encoded_files):
        if len(file_list) != n_bunches:
            raise ValueError(
                f"All encoding patterns must have same number of files. "
                f"Pattern 0 has {n_bunches}, pattern {i} has {len(file_list)}"
            )

    if S_matrix is None:
        print(f"Generating S matrix for n={n_merged_frames}...")
        S = generate_s_matrix(n_merged_frames)
    else:
        S = S_matrix
        if S.shape[0] != n_merged_frames:
            raise ValueError(f"Provided S_matrix has size {S.shape[0]}, expected {n_merged_frames}")

    print("S matrix:")
    print(S)
    print()

    Sinv = compute_hadamard_inverse(S)
    print("Inverse S matrix:")
    print(Sinv)
    print()

    skip_rows = 0
    n_cols = 0
    if file_format == "auto":
        first_file = encoded_files[0][0]
        with open(first_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    parts = line.split()
                    int(parts[0])
                    skip_rows = i
                    n_cols = len(parts)
                    break
                except (ValueError, IndexError):
                    continue

        if n_cols == 7:
            file_format = "crystfel"
            print("Detected CrystFEL format: h k l I phase sigma(I) nmeas")
            print(f"Skipping {skip_rows} header lines")
        elif n_cols == 5:
            file_format = "crystfel_simple"
            print("Detected format: h k l I sigma(I)")
            print(f"Skipping {skip_rows} header lines")
        else:
            file_format = "ccp4"
            print(f"Detected CCP4-like format with {n_cols} columns")
            print(f"Skipping {skip_rows} header lines")
        print()

    # Build the block-diagonal inverse for intensity (and sigma) columns.
    # For crystfel formats: 2 blocks (I, SIGMA).
    # For ccp4: 4 blocks (I, sigI, F, sigF).
    # When decode_amplitudes is True for crystfel formats we add 2 more blocks
    # (F, sigF) computed from the per-pattern I/SIGMA before inversion.
    if file_format in ("crystfel", "crystfel_simple"):
        n_intensity_blocks = 2  # I, SIGMA
        n_amplitude_blocks = 2 if decode_amplitudes else 0  # F, sigF
    else:
        # ccp4 already carries F/sigF columns; decode both sets together.
        n_intensity_blocks = 4
        n_amplitude_blocks = 0

    n_data_blocks = n_intensity_blocks + n_amplitude_blocks
    cooler_Sinv = block_diag(*([Sinv] * n_data_blocks))

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    decoded_bunches = []

    for bunch_idx in range(n_bunches):
        print(f"Processing bunch {bunch_idx}/{n_bunches-1}...")

        dfs = []

        for pattern_idx in range(n_merged_frames):
            file_path = encoded_files[pattern_idx][bunch_idx]

            df = pd.read_table(
                file_path,
                delimiter=r"\s+",
                header=None,
                skiprows=skip_rows,
                comment="#",
                on_bad_lines="skip",
            )

            df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()

            if file_format == "crystfel":
                df.columns = [
                    "h",
                    "k",
                    "l",
                    f"I_{pattern_idx}",
                    f"phase_{pattern_idx}",
                    f"SIGMA_{pattern_idx}",
                    f"nmeas_{pattern_idx}",
                ]
                df = df.drop([f"phase_{pattern_idx}", f"nmeas_{pattern_idx}"], axis=1)
            elif file_format == "crystfel_simple":
                df.columns = ["h", "k", "l", f"I_{pattern_idx}", f"SIGMA_{pattern_idx}"]
            else:
                df.columns = [
                    "h",
                    "k",
                    "l",
                    f"I_{pattern_idx}",
                    f"sigI_{pattern_idx}",
                    f"F_{pattern_idx}",
                    f"sigF_{pattern_idx}",
                ]

            # Amplitude conversion happens here, per pattern, before merging
            # and before inversion, so the linear transform acts on intensities.
            if decode_amplitudes and file_format in ("crystfel", "crystfel_simple"):
                df = _intensities_to_amplitudes(df, pattern_idx, file_format)

            dfs.append(df)

        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.merge(df, on=["h", "k", "l"], how="inner")

        print(f"  Found {len(combined)} common reflections")

        combined[["h", "k", "l"]] = combined[["h", "k", "l"]].astype(int)

        data_cols = [col for col in combined.columns if col not in ["h", "k", "l"]]
        combined[data_cols] = combined[data_cols].apply(pd.to_numeric, errors="coerce")

        combined = combined.dropna()

        if len(combined) == 0:
            print("  WARNING: No valid reflections after conversion!")
            continue

        # Build column order: intensities first, then amplitudes (if present).
        reordered_cols = ["h", "k", "l"]

        if file_format in ("crystfel", "crystfel_simple"):
            for col_type in ["I", "SIGMA"]:
                reordered_cols.extend([f"{col_type}_{i}" for i in range(n_merged_frames)])
            if decode_amplitudes:
                for col_type in ["F", "sigF"]:
                    reordered_cols.extend([f"{col_type}_{i}" for i in range(n_merged_frames)])
        else:
            for col_type in ["I", "sigI", "F", "sigF"]:
                reordered_cols.extend([f"{col_type}_{i}" for i in range(n_merged_frames)])

        combined = combined[reordered_cols]

        data_only = combined.drop(["h", "k", "l"], axis=1)
        decoded_data = data_only.apply(
            lambda vec: np.dot(cooler_Sinv, vec.values), axis=1, result_type="expand"
        )

        combined.iloc[:, 3:] = decoded_data.values

        # Rename decoded columns.
        if file_format in ("crystfel", "crystfel_simple"):
            intensity_cols = [
                f"{col_type}_frame{i}"
                for col_type in ["I", "SIGMA"]
                for i in range(n_merged_frames)
            ]
            amplitude_cols = (
                [
                    f"{col_type}_frame{i}"
                    for col_type in ["F", "sigF"]
                    for i in range(n_merged_frames)
                ]
                if decode_amplitudes
                else []
            )
            combined.columns = ["h", "k", "l"] + intensity_cols + amplitude_cols
        else:
            combined.columns = ["h", "k", "l"] + [
                f"{col_type}_frame{i}"
                for col_type in ["I", "sigI", "F", "sigF"]
                for i in range(n_merged_frames)
            ]

        decoded_bunches.append(combined)

        if output_dir:
            for frame_idx in range(n_merged_frames):
                _write_intensity_frame(
                    combined,
                    frame_idx,
                    bunch_idx,
                    output_path,
                    output_prefix,
                    file_format,
                    n_merged_frames,
                )
                if decode_amplitudes and file_format in ("crystfel", "crystfel_simple"):
                    _write_amplitude_frame(
                        combined,
                        frame_idx,
                        bunch_idx,
                        output_path,
                        output_prefix,
                    )

            print(f"  Saved decoded frames to {output_dir}")

    print(f"\nDecoding complete! Processed {len(decoded_bunches)} bunches.")

    return decoded_bunches


def _write_intensity_frame(
    combined: pd.DataFrame,
    frame_idx: int,
    bunch_idx: int,
    output_path: Path,
    output_prefix: str,
    file_format: str,
    n_merged_frames: int,
) -> None:
    """Write the intensity (and sigma) columns for one decoded frame.

    :param combined: Fully decoded DataFrame for the current bunch
    :param frame_idx: Index of the frame within the bunch
    :param bunch_idx: Index of the bunch
    :param output_path: Directory in which to write the file
    :param output_prefix: Filename stem prefix
    :param file_format: One of ``"crystfel"``, ``"crystfel_simple"``, or ``"ccp4"``
    :param n_merged_frames: Number of frames per bunch (used only for ccp4 column selection)
    """
    output_file = output_path / f"{output_prefix}_bunch{bunch_idx}_frame{frame_idx}.hkl"

    if file_format in ("crystfel", "crystfel_simple"):
        frame_cols = ["h", "k", "l", f"I_frame{frame_idx}", f"SIGMA_frame{frame_idx}"]
        fmt = "%4d %4d %4d %12.4f %12.4f"

        with open(output_file, "w") as f:
            f.write("CrystFEL reflection list version 2.0\n")
            f.write("Symmetry: 6/m\n")
            f.write("   h    k    l          I   sigma(I)\n")

        with open(output_file, "a") as f:
            np.savetxt(f, combined[frame_cols].values, fmt=fmt)
            f.write("End of reflections\n")

    else:
        frame_cols = [
            "h",
            "k",
            "l",
            f"I_frame{frame_idx}",
            f"sigI_frame{frame_idx}",
            f"F_frame{frame_idx}",
            f"sigF_frame{frame_idx}",
        ]
        fmt = "%4d %4d %4d %12.4f %12.4f %12.4f %12.4f"
        np.savetxt(
            output_file,
            combined[frame_cols].values,
            fmt=fmt,
            header="h    k    l    I            sigI         F            sigF",
            comments="# ",
        )


def _write_amplitude_frame(
    combined: pd.DataFrame,
    frame_idx: int,
    bunch_idx: int,
    output_path: Path,
    output_prefix: str,
) -> None:
    """Write the amplitude (F, sigF) columns for one decoded frame.

    Only called for ``crystfel`` / ``crystfel_simple`` formats when
    ``decode_amplitudes=True``; ccp4 files already include amplitudes in the
    main intensity output.

    :param combined: Fully decoded DataFrame for the current bunch
    :param frame_idx: Index of the frame within the bunch
    :param bunch_idx: Index of the bunch
    :param output_path: Directory in which to write the file
    :param output_prefix: Filename stem prefix
    """
    output_file = output_path / f"{output_prefix}_bunch{bunch_idx}_frame{frame_idx}_F.hkl"
    frame_cols = ["h", "k", "l", f"F_frame{frame_idx}", f"sigF_frame{frame_idx}"]
    fmt = "%4d %4d %4d %12.4f %12.4f"

    with open(output_file, "w") as f:
        f.write("CrystFEL reflection list version 2.0\n")
        f.write("Symmetry: 6/m\n")
        f.write("   h    k    l          F   sigma(F)\n")

    with open(output_file, "a") as f:
        np.savetxt(f, combined[frame_cols].values, fmt=fmt)
        f.write("End of reflections\n")
