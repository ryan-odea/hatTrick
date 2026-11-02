import glob

import click

from .._helpers import generate_s_matrix
from .HATRX import decode_hadamard_files
from .stream import decode_stream as _decode_stream


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
@click.option(
    "--continuous",
    type=int,
    default=None,
    help="Offset used during continuous encoding."
)
def decode(n_frames, pattern, output_dir, prefix, continuous):
    """Decode Hadamard-encoded crystallographic files.

    Example usage:

    \b
    hatrx decode -n 3 \\
        -p "data/*01-*.hkl" \\
        -p "data/*02-*.hkl" \\
        -p "data/*12-*.hkl" \\
        -o decoded_output \\
        --prefix my_data

    \b
    For continuous encoding with offset 1:
    hatrx decode -n 3 \\
        -p "data/*01-*.hkl" \\
        -p "data/*02-*.hkl" \\
        -p "data/*12-*.hkl" \\
        -o decoded_output \\
        --continuous 1
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
            continuous_offset=continuous,
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
            click.echo(f"  Pattern {i}: {row} -> frames {frames_str} summed")
        click.echo()
        click.echo("Example command:")
        click.echo(f"  hatrx decode -n {n_frames} \\")
        for i, row in enumerate(S):
            frames_summed = [j for j, val in enumerate(row) if val == 1]
            pattern_name = "".join(str(f) for f in frames_summed)
            click.echo(f'    -p "data/*{pattern_name}-*.hkl" \\')
        click.echo("    -o decoded_output")
    except Exception as e:
        raise click.ClickException(str(e))


@click.command("decode-stream")
@click.option("-n", "--n-frames", type=int, required=True, help="Number of merged frames (must be a prime number)")
@click.option("-i", "--input", "stream", type=click.Path(exists=True), required=True, help="Encoded CrystFEL stream (after ambigator)")
@click.option("-o", "--output", type=click.Path(), required=True, help="Decoded output stream")
@click.option(
    "--split",
    type=click.Path(exists=True),
    default=None,
    help="Gating file with lines '<filename> <event> <block> <pattern>'. "
    "Without it, blocks are inferred from the encoder's '{stem}_{tag}.h5' naming.",
)
@click.option("--continuous", type=int, default=None, help="Offset used during continuous encoding.")
@click.option("--symmetry", type=str, default=None,
              help="Laue group (e.g. 6/m) used to bring the n chunks of a block into one indexing setting before joining on hkl.")
@click.option("--min-common", type=int, default=1, show_default=True,
              help="Skip blocks with fewer common reflections than this.")
def decode_stream(n_frames, stream, output, split, continuous, symmetry, min_common):
    """Decode an encoded CrystFEL stream into per-frame chunks, before scaling.

    \b
    hatrx decode-stream -n 3 -i ambig.stream -o decoded.stream --split gating.txt
    partialator -i decoded.stream -o decoded -y 6/m --model=xsphere
    """
    try:
        _decode_stream(stream, output, n_frames, split_file=split, continuous_offset=continuous,
                       symmetry=symmetry, min_common=min_common)
    except Exception as e:
        raise click.ClickException(str(e))
