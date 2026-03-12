import click

from .merger import Merger


@click.command()
@click.option(
    "-f",
    "--file-name",
    required=True,
    help=(
        "Input HDF5 file, glob pattern (e.g. 'data/run_*.h5'), or path to a "
        "plain-text list file containing one HDF5 path per line.  Blocks may "
        "span file boundaries."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    default=".",
    help="Output directory for encoded HDF5 files",
)
@click.option(
    "--n-merged-frames",
    type=int,
    default=3,
    help="S-matrix order (must be prime and satisfy n == 3 mod 4)",
)
@click.option("--data-location", type=str, default="entry/data", help="HDF5 group path")
@click.option("--data-name", type=str, default="data", help="Dataset name")
@click.option(
    "--start",
    "start_index",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Global frame index (0-based) at which to begin forming blocks.  "
        "Counts from the first frame of the first resolved input file."
    ),
)
@click.option(
    "--continuous",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable continuous encoding across file boundaries. When enabled, blocks "
        "may span multiple files. When disabled (default), each file is encoded "
        "independently."
    ),
)
def encode(
    file_name: str,
    output_dir: str,
    n_merged_frames: int,
    data_location: str,
    data_name: str,
    start_index: int,
    continuous: bool,
) -> None:
    """Hadamard-encode HDF5 frames across one or more input files.

    Output files are named after the dominant input file with the binary
    S-matrix row appended (e.g. input_000_110.h5 for n=3).  By default,
    each file is encoded independently. Use --continuous to allow blocks
    to span file boundaries.

    :param file_name: Input HDF5 file, glob pattern, or path to plain-text list file
    :param output_dir: Output directory for encoded HDF5 files
    :param n_merged_frames: S-matrix order (must be prime and satisfy n == 3 mod 4)
    :param data_location: HDF5 group path (e.g., "entry/data")
    :param data_name: Dataset name within the HDF5 group
    :param start_index: Global frame index (0-based) at which to begin forming blocks
    :param continuous: If True, blocks may span file boundaries; if False, each file is processed independently

    \b
    Examples
    --------
    Single file:

    \b
        hatrx encode -f run.h5 -o encoded/ --n-merged-frames 7

    Many files via glob (no shell ARG_MAX limit):

    \b
        hatrx encode -f "data/run_*.h5" -o merged/ --n-merged-frames 3

    Many files via list file, starting at frame 100:

    \b
        hatrx encode -f files.txt -o merged/ --n-merged-frames 3 --start 100
    """
    merger = Merger(
        file_name=file_name,
        output_dir=output_dir,
        n_merged_frames=n_merged_frames,
        data_location=data_location,
        data_name=data_name,
        start=start_index,
        continuous=continuous,
    )
    merger.process()
