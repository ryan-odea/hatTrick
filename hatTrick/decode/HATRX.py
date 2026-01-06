import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from io import StringIO
from typing import List, Optional
from pathlib import Path

from ._helpers import compute_hadamard_inverse, generate_s_matrix


def decode_hadamard_files(
    encoded_files: List[List[str]],
    n_merged_frames: int = 3,
    output_dir: Optional[str] = None,
    output_prefix: str = "decoded",
    S_matrix: Optional[np.ndarray] = None,
    file_format: str = "auto",
) -> List[pd.DataFrame]:

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
            print(f"Detected CrystFEL format: h k l I phase sigma(I) nmeas")
            print(f"Skipping {skip_rows} header lines")
        elif n_cols == 5:
            file_format = "crystfel_simple"
            print(f"Detected format: h k l I sigma(I)")
            print(f"Skipping {skip_rows} header lines")
        else:
            file_format = "ccp4"
            print(f"Detected CCP4-like format with {n_cols} columns")
            print(f"Skipping {skip_rows} header lines")
        print()

    if file_format == "crystfel":
        data_columns = ["I", "SIGMA"]
        n_data_blocks = 2
    elif file_format == "crystfel_simple":
        data_columns = ["I", "SIGMA"]
        n_data_blocks = 2
    else:
        data_columns = ["I", "sigI", "F", "sigF"]
        n_data_blocks = 4

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
            print(f"  WARNING: No valid reflections after conversion!")
            continue

        reordered_cols = ["h", "k", "l"]

        if file_format in ["crystfel", "crystfel_simple"]:
            for col_type in ["I", "SIGMA"]:
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

        if file_format in ["crystfel", "crystfel_simple"]:
            combined.columns = ["h", "k", "l"] + [
                f"{col_type}_frame{i}"
                for col_type in ["I", "SIGMA"]
                for i in range(n_merged_frames)
            ]
        else:
            combined.columns = ["h", "k", "l"] + [
                f"{col_type}_frame{i}"
                for col_type in ["I", "sigI", "F", "sigF"]
                for i in range(n_merged_frames)
            ]

        decoded_bunches.append(combined)

        if output_dir:
            for frame_idx in range(n_merged_frames):
                output_file = output_path / f"{output_prefix}_bunch{bunch_idx}_frame{frame_idx}.hkl"

                if file_format in ["crystfel", "crystfel_simple"]:
                    frame_cols = ["h", "k", "l", f"I_frame{frame_idx}", f"SIGMA_frame{frame_idx}"]
                    fmt = "%4d %4d %4d %12.4f %12.4f"

                    with open(output_file, "w") as f:
                        f.write("CrystFEL reflection list version 2.0\n")
                        f.write("Symmetry: 6/m\n")
                        f.write("   h    k    l          I   sigma(I)\n")

                    frame_data = combined[frame_cols]
                    with open(output_file, "a") as f:
                        np.savetxt(f, frame_data.values, fmt=fmt)
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

                    frame_data = combined[frame_cols]
                    np.savetxt(
                        output_file,
                        frame_data.values,
                        fmt=fmt,
                        header="h    k    l    I            sigI         F            sigF",
                        comments="# ",
                    )

            print(f"  Saved decoded frames to {output_dir}")

    print(f"\nDecoding complete! Processed {len(decoded_bunches)} bunches.")

    return decoded_bunches
