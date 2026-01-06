import click
from .encode.cli import encode
from .decode.cli import decode, show_encoding


@click.group()
@click.version_option()
def cli():
    """hatTrick (HATRX) - Hadamard encoding/decoding for Time-Resolved Crystallography."""
    pass


cli.add_command(encode)
cli.add_command(decode)
cli.add_command(show_encoding)


if __name__ == "__main__":
    cli()