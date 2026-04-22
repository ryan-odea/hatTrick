"""Post-decode resolution-shell scaling for Hadamard-decoded HKL files."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl


def read_crystfel_hkl(path: str) -> Tuple[pl.DataFrame, List[str], str]:
    """Parse a CrystFEL HKL file.

    Returns (data_df, header_lines, symmetry).
    The header_lines include everything before the data block and the
    'End of reflections' footer, so the file can be round-tripped.
    """
    path = str(path)
    header_lines: List[str] = []
    data_rows: List[List[str]] = []
    symmetry = ""
    footer_lines: List[str] = []
    in_data = False
    past_data = False

    with open(path) as fh:
        for line in fh:
            if past_data:
                footer_lines.append(line)
                continue

            stripped = line.strip()

            if not in_data:
                header_lines.append(line)
                if stripped.startswith("Symmetry:"):
                    symmetry = stripped.split(":", 1)[1].strip()
                # Detect start of data: first line with an integer first token
                # after header lines
                if stripped and not stripped.startswith("#") and not stripped.startswith("CrystFEL") and not stripped.startswith("Symmetry"):
                    parts = stripped.split()
                    try:
                        int(parts[0])
                        # This is a data line — remove it from header
                        header_lines.pop()
                        in_data = True
                        data_rows.append(parts)
                    except (ValueError, IndexError):
                        pass
            else:
                if stripped.lower().startswith("end of"):
                    past_data = True
                    footer_lines.append(line)
                    continue
                parts = stripped.split()
                if parts:
                    try:
                        int(parts[0])
                        data_rows.append(parts)
                    except (ValueError, IndexError):
                        pass

    if not data_rows:
        raise ValueError(f"No data rows found in {path}")

    n_cols = len(data_rows[0])
    # CrystFEL full format: h k l I phase sigma nmeas
    # CrystFEL simple:      h k l I sigma
    if n_cols == 7:
        cols = ["h", "k", "l", "I", "phase", "sigma", "nmeas"]
        dtypes = {
            "h": pl.Int32, "k": pl.Int32, "l": pl.Int32,
            "I": pl.Float64, "phase": pl.Float64,
            "sigma": pl.Float64, "nmeas": pl.Int32,
        }
    elif n_cols == 5:
        cols = ["h", "k", "l", "I", "sigma"]
        dtypes = {
            "h": pl.Int32, "k": pl.Int32, "l": pl.Int32,
            "I": pl.Float64, "sigma": pl.Float64,
        }
    else:
        raise ValueError(f"Unexpected column count {n_cols} in {path}")

    # Build DataFrame
    raw = {c: [row[i] for row in data_rows] for i, c in enumerate(cols)}
    df = pl.DataFrame(raw).cast(dtypes)

    # Append footer to header for round-trip
    header_lines.extend(footer_lines)

    return df, header_lines, symmetry


def write_crystfel_hkl(path: str, df: pl.DataFrame, header_lines: List[str]) -> None:
    """Write a CrystFEL HKL file preserving original header/footer."""
    path = str(path)
    # Split header_lines into pre-data header and post-data footer
    pre_data: List[str] = []
    post_data: List[str] = []
    found_end = False
    for line in header_lines:
        if line.strip().lower().startswith("end of"):
            found_end = True
        if found_end:
            post_data.append(line)
        else:
            pre_data.append(line)

    cols = df.columns  # e.g. h k l I sigma  or  h k l I phase sigma nmeas
    has_phase = "phase" in cols

    with open(path, "w") as fh:
        for line in pre_data:
            fh.write(line)

        for row in df.iter_rows():
            if has_phase:
                h, k, l, I, phase, sigma, nmeas = row[:7]
                fh.write(f"{h:4d} {k:4d} {l:4d} {I:12.4f} {phase:12.4f} {sigma:12.4f} {nmeas:7d}\n")
            else:
                h, k, l, I, sigma = row[:5]
                fh.write(f"{h:4d} {k:4d} {l:4d} {I:12.4f} {sigma:12.4f}\n")

        for line in post_data:
            fh.write(line)


def compute_resolution(
    h: np.ndarray, k: np.ndarray, l: np.ndarray,
    cell: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Compute 1/d^2 for each reflection using the general triclinic formula.

    Parameters
    ----------
    h, k, l : arrays of Miller indices
    cell : (a, b, c, alpha_deg, beta_deg, gamma_deg)

    Returns
    -------
    inv_d_sq : 1/d^2 array
    """
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)

    V = a * b * c * np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
    V2 = V**2

    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    inv_d_sq = (1.0 / V2) * (
        h**2 * b**2 * c**2 * sa**2
        + k**2 * a**2 * c**2 * sb**2
        + l**2 * a**2 * b**2 * sg**2
        + 2.0 * h * k * a * b * c**2 * (ca * cb - cg)
        + 2.0 * k * l * a**2 * b * c * (cb * cg - ca)
        + 2.0 * h * l * a * b**2 * c * (ca * cg - cb)
    )
    return inv_d_sq


def compute_shell_scales(
    df_merged: pl.DataFrame,
    n_shells: int = 20,
    poly_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-shell scale factors and fit a polynomial.

    Parameters
    ----------
    df_merged : DataFrame with columns 'I_dark', 'I_dec', 'inv_d_sq'
    n_shells : number of resolution shells
    poly_order : polynomial order for fit to k vs 1/d^2

    Returns
    -------
    poly_coeffs : numpy poly1d coefficients (highest power first)
    diagnostics : array of shape (n_shells, 3) with columns
                  [shell_center, k_shell, n_refl]
    """
    inv_d_sq = df_merged["inv_d_sq"].to_numpy()
    I_dark = df_merged["I_dark"].to_numpy()
    I_dec = df_merged["I_dec"].to_numpy()

    s_min, s_max = inv_d_sq.min(), inv_d_sq.max()
    edges = np.linspace(s_min, s_max, n_shells + 1)

    centers = []
    k_shells = []
    n_refls = []

    for i in range(n_shells):
        mask = (inv_d_sq >= edges[i]) & (inv_d_sq < edges[i + 1])
        if i == n_shells - 1:
            mask = (inv_d_sq >= edges[i]) & (inv_d_sq <= edges[i + 1])

        n = mask.sum()
        if n == 0:
            continue

        id_shell = I_dark[mask]
        ie_shell = I_dec[mask]

        denom = np.sum(ie_shell**2)
        if denom == 0:
            continue

        k = np.sum(id_shell * ie_shell) / denom
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        k_shells.append(k)
        n_refls.append(n)

    centers = np.array(centers)
    k_shells = np.array(k_shells)
    n_refls = np.array(n_refls)

    if len(centers) < poly_order + 1:
        raise ValueError(
            f"Only {len(centers)} non-empty shells; need at least {poly_order + 1} "
            f"for a degree-{poly_order} polynomial fit."
        )

    poly_coeffs = np.polyfit(centers, k_shells, poly_order)

    diagnostics = np.column_stack([centers, k_shells, n_refls])
    return poly_coeffs, diagnostics


def apply_scale(df: pl.DataFrame, poly_coeffs: np.ndarray) -> pl.DataFrame:
    """Apply polynomial scale factor k(1/d^2) to I and sigma columns.

    Parameters
    ----------
    df : DataFrame with columns 'I', 'sigma', 'inv_d_sq'
    poly_coeffs : polynomial coefficients from np.polyfit

    Returns
    -------
    Scaled DataFrame (new columns overwrite I and sigma).
    """
    inv_d_sq = df["inv_d_sq"].to_numpy()
    k = np.polyval(poly_coeffs, inv_d_sq)

    return df.with_columns(
        (pl.col("I") * pl.Series(k)).alias("I"),
        (pl.col("sigma") * pl.Series(np.abs(k))).alias("sigma"),
    )


def scale_hkl_files(
    dark_path: str,
    decoded_paths: List[str],
    cell: Tuple[float, float, float, float, float, float],
    spacegroup: str = "P1",
    n_shells: int = 20,
    poly_order: int = 3,
    output_dir: str = ".",
    prefix: str = "scaled",
    diagnostics: bool = False,
) -> List[pl.DataFrame]:
    """Scale decoded HKL files against a dark/ground-state reference.

    Parameters
    ----------
    dark_path : path to dark/reference CrystFEL HKL
    decoded_paths : list of paths to decoded CrystFEL HKL files
    cell : (a, b, c, alpha, beta, gamma) in Angstrom/degrees
    spacegroup : space group symbol for output header
    n_shells : number of resolution shells
    poly_order : polynomial fit order
    output_dir : directory for output files
    prefix : filename prefix for scaled outputs
    diagnostics : if True, print per-shell diagnostics

    Returns
    -------
    List of scaled DataFrames
    """
    dark_df, dark_header, _ = read_crystfel_hkl(dark_path)

    # Compute resolution for dark
    inv_d_sq_dark = compute_resolution(
        dark_df["h"].to_numpy(), dark_df["k"].to_numpy(), dark_df["l"].to_numpy(),
        cell,
    )
    dark_with_res = dark_df.with_columns(pl.Series("inv_d_sq", inv_d_sq_dark))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, dec_path in enumerate(decoded_paths):
        dec_df, dec_header, _ = read_crystfel_hkl(dec_path)

        # Inner join on h, k, l
        merged = dark_with_res.select(["h", "k", "l", "I", "inv_d_sq"]).rename({"I": "I_dark"}).join(
            dec_df.select(["h", "k", "l", "I", "sigma"]).rename({"I": "I_dec", "sigma": "sigma_dec"}),
            on=["h", "k", "l"],
            how="inner",
        )

        if merged.height == 0:
            raise ValueError(
                f"No overlapping reflections between dark and {dec_path}"
            )

        poly_coeffs, diag = compute_shell_scales(
            merged.rename({"I_dec": "I_dec_tmp"}).with_columns(
                pl.col("I_dec_tmp").alias("I_dec")
            ).select(["I_dark", "I_dec", "inv_d_sq"]),
            n_shells=n_shells,
            poly_order=poly_order,
        )

        if diagnostics:
            print(f"\n--- Diagnostics for {Path(dec_path).name} ---")
            print(f"{'Shell center (1/d²)':>22s}  {'k_shell':>10s}  {'n_refl':>8s}")
            for row in diag:
                print(f"{row[0]:22.6f}  {row[1]:10.4f}  {int(row[2]):8d}")
            print(f"Polynomial coefficients (degree {poly_order}): {poly_coeffs}")

        # Apply scale to the full decoded file (not just the overlap)
        inv_d_sq_dec = compute_resolution(
            dec_df["h"].to_numpy(), dec_df["k"].to_numpy(), dec_df["l"].to_numpy(),
            cell,
        )
        dec_with_res = dec_df.with_columns(pl.Series("inv_d_sq", inv_d_sq_dec))

        scaled = apply_scale(dec_with_res, poly_coeffs)

        # Drop the inv_d_sq helper column for output
        scaled_out = scaled.drop("inv_d_sq")

        # Rebuild header with correct symmetry
        out_header = []
        footer = []
        in_footer = False
        for line in dec_header:
            if line.strip().lower().startswith("end of"):
                in_footer = True
            if in_footer:
                footer.append(line)
            else:
                if line.strip().startswith("Symmetry:"):
                    out_header.append(f"Symmetry: {spacegroup}\n")
                else:
                    out_header.append(line)
        out_header.extend(footer)

        out_path = out_dir / f"{prefix}_{Path(dec_path).stem}.hkl"
        write_crystfel_hkl(str(out_path), scaled_out, out_header)

        print(f"Scaled {Path(dec_path).name} -> {out_path.name} "
              f"({scaled_out.height} reflections)")

        results.append(scaled_out)

    return results
