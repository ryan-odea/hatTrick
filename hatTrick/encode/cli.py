import click

from .merger import Merger


@click.command()
@click.option("-f", "--file-name", required=True, help="Input HDF5 file")
@click.option("-o", "--output-file", default="merged.h5", help="Output HDF5 file")
@click.option("--n-frames", type=int, default=10000, help="Number of frames to read")
@click.option(
    "--n-merged-frames",
    type=int,
    default=3,
    help="Number of frames per block (Hadamard: must be prime and ≡ 3 mod 4)",
)
@click.option(
    "--skip-pattern",
    type=str,
    default=None,
    help="Comma-separated indices to skip (rolling merge only)",
)
@click.option(
    "--type",
    "merge_type",
    type=click.Choice(["hadamard", "rolling"]),
    default="hadamard",
    help="Merge type: 'hadamard' (proper encoding) or 'rolling' (simple merge)",
)
@click.option("--data-location", type=str, default="entry/data", help="HDF5 group path")
@click.option("--data-name", type=str, default="data", help="Dataset name")
@click.option("--n-workers", type=int, default=None, help="Number of parallel workers")
@click.option("--sequential", is_flag=True, help="Run sequentially instead of parallel")
def encode(
    file_name,
    output_file,
    n_frames,
    n_merged_frames,
    skip_pattern,
    merge_type,
    data_location,
    data_name,
    n_workers,
    sequential,
):
    """Encode HDF5 frames with Hadamard encoding or rolling merge."""
    parsed_skip_pattern = None
    if skip_pattern:
        parsed_skip_pattern = [int(x.strip()) for x in skip_pattern.split(",")]

    if merge_type == "hadamard" and parsed_skip_pattern is not None:
        click.echo("Warning: skip_pattern ignored for Hadamard encoding")
        parsed_skip_pattern = None

    merger = Merger(
        file_name=file_name,
        output_file=output_file,
        n_frames=n_frames,
        n_merged_frames=n_merged_frames,
        skip_pattern=parsed_skip_pattern,
        data_location=data_location,
        data_name=data_name,
        n_workers=n_workers,
        type=merge_type,
    )
    merger.process(parallel=not sequential)
