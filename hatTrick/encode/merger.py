from .._helpers import (
    _validate_hadamard,
    continuous_hadamard_encode,
    resolve_file_list,
    single_file_hadamard_encode,
)


class Merger:
    """Hadamard-encode HDF5 detector frames across one or more input files.

    Frames are streamed via :func:`_frame_iterator` so that at most one block
    of *n_merged_frames* frames is held in RAM at any given time, regardless of
    how many input files are provided or how large they are.

    In continuous mode, blocks may span file boundaries: when the S-matrix window
    extends past the end of one file, the remaining frames are drawn from the next
    file. In non-continuous mode (default), each file is processed independently.

    Output files are named after the dominant input file stem with the binary
    S-matrix row appended (e.g. ``input_000_110.h5`` for n=3).

    :param file_name: Path to a single HDF5 file, a glob pattern (e.g. ``"data/run_*.h5"``), or a plain-text list file containing one HDF5 path per line; see :func:`resolve_file_list` for resolution rules
    :param output_dir: Directory for output HDF5 files
    :param n_merged_frames: S-matrix order; must be a prime satisfying n % 4 == 3
    :param data_location: HDF5 group path (e.g. ``"entry/data"``)
    :param data_name: Dataset name inside *data_location*
    :param start: Global frame index (0-based, counting from the first frame of the first resolved file) at which to begin forming blocks
    :param continuous: If True, blocks may span file boundaries; if False, each file is processed independently
    """

    def __init__(
        self,
        file_name: str,
        output_dir: str = ".",
        n_merged_frames: int = 3,
        data_location: str = "entry/data",
        data_name: str = "data",
        start: int = 0,
        continuous: bool = False,
    ):
        self.file_name = file_name
        self.output_dir = output_dir
        self.n_merged_frames = n_merged_frames
        self.data_location = data_location
        self.data_name = data_name
        self.start = start
        self.continuous = continuous

    def process(self) -> None:
        """Resolve input files and run the Hadamard encode."""
        _validate_hadamard(self.n_merged_frames)

        file_paths = resolve_file_list(self.file_name)

        if self.continuous:
            continuous_hadamard_encode(
                file_paths=file_paths,
                n_merged_frames=self.n_merged_frames,
                data_location=self.data_location,
                data_name=self.data_name,
                output_dir=self.output_dir,
                start_index=self.start,
            )
        else:
            single_file_hadamard_encode(
                file_paths=file_paths,
                n_merged_frames=self.n_merged_frames,
                data_location=self.data_location,
                data_name=self.data_name,
                output_dir=self.output_dir,
                start_index=self.start,
            )
