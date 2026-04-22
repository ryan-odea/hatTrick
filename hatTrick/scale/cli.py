"""CLI subcommand for post-decode resolution-shell scaling."""

from typing import Tuple

import click

from .scaler import scale_hkl_files


def _parse_cell(cell_str: str) -> Tuple[float, float, float, float, float, float]:
    """Parse 'a b c alpha beta gamma' into a 6-tuple of floats."""
    parts = cell_str.split()
    if len(parts) != 6:
        raise click.BadParameter(
            f"Expected 6 values (a b c alpha beta gamma), got {len(parts)}"
        )
    try:
        values = [float(x) for x in parts]
    except ValueError as e:
        raise click.BadParameter(f"Non-numeric cell parameter: {e}")
    return (values[0], values[1], values[2], values[3], values[4], values[5])


@click.command()
@click.option(
    "--dark",
    required=True,
    type=click.Path(exists=True),
    help="Dark/ground-state reference HKL file.",
)
@click.option(
    "--cell",
    required=True,
    type=str,
    help="Unit cell: 'a b c alpha beta gamma' (Angstrom / degrees).",
)
@click.option(
    "--spacegroup",
    required=True,
    type=str,
    help="Space group symbol for output header.",
)
@click.option(
    "--n-shells",
    default=20,
    type=int,
    show_default=True,
    help="Number of resolution shells.",
)
@click.option(
    "--poly-order",
    default=3,
    type=int,
    show_default=True,
    help="Polynomial order for scale-factor fit.",
)
@click.option(
    "-o",
    "--output-dir",
    default=".",
    type=click.Path(),
    show_default=True,
    help="Output directory for scaled files.",
)
@click.option(
    "--prefix",
    default="scaled",
    type=str,
    show_default=True,
    help="Prefix for output filenames.",
)
@click.option(
    "--diagnostics",
    is_flag=True,
    default=False,
    help="Print per-shell scaling diagnostics.",
)
@click.argument("decoded_files", nargs=-1, required=True, type=click.Path(exists=True))
def scale(dark, cell, spacegroup, n_shells, poly_order, output_dir, prefix, diagnostics, decoded_files):
    """Scale decoded HKL files against a dark/ground-state reference.

    Applies resolution-dependent scaling after Hadamard decoding.

    \b
    Example:
        hatrx scale --dark dark.hkl \\
            --cell '78.0 78.0 37.0 90 90 120' \\
            --spacegroup P6322 \\
            --n-shells 20 --poly-order 3 \\
            -o scaled_output/ \\
            --diagnostics \\
            decoded_frame*.hkl
    """
    try:
        cell_params = _parse_cell(cell)
    except click.BadParameter as e:
        raise click.UsageError(str(e))

    try:
        scale_hkl_files(
            dark_path=dark,
            decoded_paths=list(decoded_files),
            cell=cell_params,
            spacegroup=spacegroup,
            n_shells=n_shells,
            poly_order=poly_order,
            output_dir=output_dir,
            prefix=prefix,
            diagnostics=diagnostics,
        )
    except Exception as e:
        raise click.ClickException(str(e))
