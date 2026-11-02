from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .._helpers import compute_hadamard_inverse, generate_s_matrix, hadamard_decode


def decode_hadamard_files(
    encoded_files: List[List[str]],
    n_merged_frames: int = 3,
    output_dir: Optional[str] = None,
    output_prefix: str = "decoded",
    S_matrix: Optional[np.ndarray] = None,
    file_format: str = "auto",
    continuous_offset: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Decode scaled CrystFEL ``.hkl`` files, one list of files per encoding pattern.

    :param encoded_files: ``n_merged_frames`` lists of paths, each sorted by bunch
    :param n_merged_frames: S-matrix order
    :param output_dir: Where to write ``{output_prefix}_frame{i}.hkl``; nothing is written when ``None``
    :param S_matrix: Pre-computed S-matrix; generated when ``None``
    :param file_format: ``"auto"``, ``"crystfel"`` (7 columns) or ``"crystfel_simple"`` (5 columns)
    :param continuous_offset: Offset used during continuous encoding
    :returns: One DataFrame per bunch with decoded ``I_frame{i}`` and ``SIGMA_frame{i}`` columns
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
            raise ValueError(
                f"Unsupported .hkl layout with {n_cols} columns; expected 5 (h k l I sigma) or 7 (CrystFEL)"
            )
        print()

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
            else:
                df.columns = ["h", "k", "l", f"I_{pattern_idx}", f"SIGMA_{pattern_idx}"]

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

        n = n_merged_frames
        I_cols = [f"I_{i}" for i in range(n)]
        sig_cols = [f"SIGMA_{i}" for i in range(n)]
        combined = combined[["h", "k", "l"] + I_cols + sig_cols]
        I_dec, sig_dec = hadamard_decode(combined[I_cols].values, combined[sig_cols].values, Sinv)
        combined.iloc[:, 3 : 3 + n] = I_dec
        combined.iloc[:, 3 + n :] = sig_dec
        combined.columns = ["h", "k", "l"] + [
            f"{col_type}_frame{i}" for col_type in ["I", "SIGMA"] for i in range(n)
        ]

        decoded_bunches.append(combined)

        if output_dir:
            for frame_idx in range(n_merged_frames):
                if continuous_offset is not None:
                    corrected_idx = (frame_idx - continuous_offset) % n_merged_frames
                else:
                    corrected_idx = frame_idx
                global_frame = bunch_idx * n_merged_frames + corrected_idx
                output_file = output_path / f"{output_prefix}_frame{global_frame}.hkl"

                frame_cols = ["h", "k", "l", f"I_frame{frame_idx}", f"SIGMA_frame{frame_idx}"]
                with open(output_file, "w") as f:
                    f.write("CrystFEL reflection list version 2.0\n")
                    f.write("Symmetry: 6/m\n")
                    f.write("   h    k    l          I   sigma(I)\n")
                    np.savetxt(f, combined[frame_cols].values, fmt="%4d %4d %4d %12.4f %12.4f")
                    f.write("End of reflections\n")

            print(f"  Saved decoded frames to {output_dir}")

    print(f"\nDecoding complete! Processed {len(decoded_bunches)} bunches.")

    return decoded_bunches
