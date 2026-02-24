from typing import Tuple

import click

from .._helpers import generate_s_matrix, resolve_file_list
from ..HATRX import decode_hadamard_files


@click.command()
@click.option(
    "-n",
    "--n-frames",
    type=int,
    required=True,
    help="Number of merged frames (must be a prime number)",
)
@click.option(
    "-p",
    "--pattern",
    type=str,
    multiple=True,
    help=(
        "File pattern or list-file for each encoding pattern.  Accepts a glob "
        "pattern (e.g. 'data/*_110.*.hkl') or a path to a plain-text file "
        "containing one path per line.  Provide one -p per encoding pattern, "
        "in the order given by 'hatrx show-encoding'."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Output directory for decoded files",
)
@click.option("--prefix", type=str, default="decoded", help="Prefix for output filenames")
@click.option(
    "--amplitudes",
    is_flag=True,
    default=False,
    help=(
        "Also decode structure-factor amplitudes.  Intensities are converted "
        "to amplitudes via F = sqrt(|I|) and sigF = sigI / (2*F) per encoding "
        "pattern BEFORE the Hadamard inversion (the physically correct order). "
        "For crystfel/crystfel_simple formats, amplitude-decoded frames are "
        "written to separate output files with a '_F' suffix alongside the "
        "standard intensity output files.  For ccp4 format, F/sigF are already "
        "present and decoded together with intensities in all cases."
    ),
)
def decode(
    n_frames: int,
    pattern: Tuple[str, ...],
    output_dir: str,
    prefix: str,
    amplitudes: bool,
) -> None:
    """Decode Hadamard-encoded crystallographic files.

    Each -p pattern resolves (via glob or list-file) to one file per bunch.
    All patterns must resolve to the same number of files.

    :param n_frames: Number of merged frames (must be a prime number)
    :param pattern: File pattern or list-file for each encoding pattern
    :param output_dir: Output directory for decoded files
    :param prefix: Prefix for output filenames
    :param amplitudes: Whether to also decode structure-factor amplitudes

    \b
    Example usage:

    \b
    hatrx decode -n 3 \\
        -p "data/*merged-*.110.*.cif" \\
        -p "data/*merged-*.101.*.cif" \\
        -p "data/*merged-*.011.*.cif" \\
        -o decoded_output \\
        --prefix my_data

    \b
    To also decode amplitudes:

    \b
    hatrx decode -n 3 \\
        -p "data/*merged-*.110.*.cif" \\
        -p "data/*merged-*.101.*.cif" \\
        -p "data/*merged-*.011.*.cif" \\
        -o decoded_output \\
        --amplitudes
    """
    if len(pattern) != n_frames:
        raise click.UsageError(
            f"Number of patterns ({len(pattern)}) must match n-frames ({n_frames})"
        )

    encoded_files = []
    for i, pat in enumerate(pattern):
        try:
            files = resolve_file_list(pat)
        except ValueError as e:
            raise click.FileError(pat, str(e))
        encoded_files.append(files)

    n_bunches = len(encoded_files[0])
    for i, files in enumerate(encoded_files[1:], 1):
        if len(files) != n_bunches:
            raise click.UsageError(
                f"Pattern {i} resolved to {len(files)} files, but pattern 0 "
                f"resolved to {n_bunches}. All patterns must have the same "
                f"number of files."
            )

    try:
        decode_hadamard_files(
            encoded_files=encoded_files,
            n_merged_frames=n_frames,
            output_dir=output_dir,
            output_prefix=prefix,
            decode_amplitudes=amplitudes,
        )
    except Exception as e:
        raise click.ClickException(str(e))


@click.command()
@click.option("-n", "--n-frames", type=int, required=True, help="Number of merged frames")
def show_encoding(n_frames: int) -> None:
    """Show the encoding patterns for a given n.

    This helps you determine the correct order for your file patterns.

    :param n_frames: Number of merged frames (must be a prime number)

    \b
    Example usage:

    \b
    hatrx show-encoding -n 3
    """
    try:
        S = generate_s_matrix(n_frames)
        click.echo(f"S matrix for n={n_frames}:")
        click.echo(S)
        click.echo()
        click.echo("Encoding patterns (provide patterns in this order):")
        for i, row in enumerate(S):
            frames_summed = [j for j, val in enumerate(row) if val == 1]
            frames_str = "+".join(str(f) for f in frames_summed)
            click.echo(f"  Pattern {i}: {row} -> frames {frames_str} summed")
        click.echo()
        click.echo("Example command:")
        click.echo(f"  hatrx decode -n {n_frames} \\")
        for i, row in enumerate(S):
            frames_summed = [j for j, val in enumerate(row) if val == 1]
            pattern_name = "".join(str(f) for f in frames_summed)
            click.echo(f'    -p "data/*merged-*.{pattern_name}.*.cif" \\')
        click.echo("    -o decoded_output")
    except Exception as e:
        raise click.ClickException(str(e))
