import click
import glob
from .HATRX import decode_hadamard_files
from ._helpers import generate_s_matrix


@click.command()
@click.option(
    "-n",
    "--n-frames",
    type=int,
    required=True,
    help="Number of merged frames (must be a prime number)",
)
@click.option("-p", "--pattern", type=str, multiple=True, help="File pattern for each encoding")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Output directory for decoded files",
)
@click.option("--prefix", type=str, default="decoded", help="Prefix for output filenames")
def decode(n_frames, pattern, output_dir, prefix):
    """Decode Hadamard-encoded crystallographic files.

    Example usage:

    \b
    hatrx decode -n 3 \\
        -p "data/*merged-*.01.*.cif" \\
        -p "data/*merged-*.02.*.cif" \\
        -p "data/*merged-*.12.*.cif" \\
        -o decoded_output \\
        --prefix my_data
    """
    if len(pattern) != n_frames:
        raise click.UsageError(
            f"Number of patterns ({len(pattern)}) must match n-frames ({n_frames})"
        )

    encoded_files = []
    for i, pat in enumerate(pattern):
        files = sorted(glob.glob(pat))
        if not files:
            raise click.FileError(pat, f"No files found matching pattern {i}")
        encoded_files.append(files)

    n_bunches = len(encoded_files[0])
    for i, files in enumerate(encoded_files[1:], 1):
        if len(files) != n_bunches:
            raise click.UsageError(
                f"Pattern {i} has {len(files)} files, but pattern 0 has {n_bunches}. "
                "All patterns must have the same number of files."
            )

    try:
        decode_hadamard_files(
            encoded_files=encoded_files,
            n_merged_frames=n_frames,
            output_dir=output_dir,
            output_prefix=prefix,
        )
    except Exception as e:
        raise click.ClickException(str(e))


@click.command()
@click.option("-n", "--n-frames", type=int, required=True, help="Number of merged frames")
def show_encoding(n_frames):
    """Show the encoding patterns for a given n.

    This helps you determine the correct order for your file patterns.

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
            click.echo(f"  Pattern {i}: {row} → frames {frames_str} summed")
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