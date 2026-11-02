import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._helpers import compute_hadamard_inverse, generate_s_matrix, hadamard_decode

_CHUNK_BEGIN = "----- Begin chunk -----"
_CHUNK_END = "----- End chunk -----"
_REFL_BEGIN = "Reflections measured after indexing"
_REFL_END = "End of reflections"
_REFL_FMT = "{:4d} {:4d} {:4d} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:6.1f} {:6.1f} {}\n"


def read_split_file(path: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Read a gating file mapping each encoded chunk to its block and pattern.

    One chunk per line, whitespace separated::

        <image filename> <event> <block id> <pattern>

    *event* is the stream's ``Event:`` value without the leading ``//`` (use
    ``-`` for streams without events).  *pattern* is either the S-matrix row
    index (``0``..``n-1``) or its binary tag (e.g. ``110``).  Blank lines and
    lines starting with ``#`` are ignored.

    :returns: Mapping ``(filename, event) -> (block, pattern)``
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 4 columns in split file, got: {line!r}")
            fname, event, block, pattern = parts
            mapping[(fname, event)] = (block, pattern)
    return mapping


def _parse_chunk(text: str) -> dict:
    fname = re.search(r"^Image filename: (.*)$", text, re.M)
    event = re.search(r"^Event: //(.*)$", text, re.M)
    crystals = re.findall(r"--- Begin crystal\n(.*?)--- End crystal", text, re.S)
    reflections = []
    for crystal in crystals:
        refl: Dict[Tuple[int, int, int], Tuple[float, float, str]] = {}
        body = crystal.split(_REFL_BEGIN + "\n", 1)
        if len(body) == 2:
            lines = body[1].split(_REFL_END, 1)[0].splitlines()[1:]  # skip column header
            for line in lines:
                p = line.split()
                if len(p) < 5:
                    continue
                refl[(int(p[0]), int(p[1]), int(p[2]))] = (float(p[3]), float(p[4]), line)
        reflections.append(refl)
    return {
        "text": text,
        "filename": fname.group(1).strip() if fname else "",
        "event": event.group(1).strip() if event else "-",
        "reflections": reflections,
    }


def _render_chunk(template: str, block: str, frame: int, serial: int, lines: List[str]) -> str:
    text = re.sub(r"^Image filename: .*$", f"Image filename: {block}", template, flags=re.M)
    text = re.sub(r"^Event: .*$", f"Event: //{frame}", text, flags=re.M)
    text = re.sub(r"^Image serial number: .*$", f"Image serial number: {serial}", text, flags=re.M)
    text = re.sub(r"^num_reflections = .*$", f"num_reflections = {len(lines)}", text, flags=re.M)
    head, rest = text.split(_REFL_BEGIN + "\n", 1)
    header_line, rest = rest.split("\n", 1)
    tail = rest.split(_REFL_END, 1)[1]
    return head + _REFL_BEGIN + "\n" + header_line + "\n" + "".join(lines) + _REFL_END + tail



_SUPPORTED_LAUE = ("6/m",)


def _laue_ops(symmetry):
    """hkl operators of the Laue group (12 for 6/m); identity only when *symmetry* is None.

    Used to bring the n chunks of a block into one indexing setting: a hexagonal
    lattice may be indexed in any 6/m-equivalent setting, which ambigator does
    not unify (it only resolves the merohedral twin ambiguity).
    """
    if symmetry is None:
        return [lambda h, k, l: (h, k, l)]
    if symmetry not in _SUPPORTED_LAUE:
        raise ValueError(f"symmetry {symmetry!r} not supported (have: {_SUPPORTED_LAUE})")
    rot = [lambda h, k, l: (h, k, l), lambda h, k, l: (-k, h + k, l), lambda h, k, l: (-h - k, h, l),
           lambda h, k, l: (-h, -k, l), lambda h, k, l: (k, -h - k, l), lambda h, k, l: (h + k, -h, l)]
    return rot + [(lambda op: (lambda h, k, l: op(-h, -k, -l)))(op) for op in rot]


def _align_to(ref: dict, refl: dict, ops) -> Tuple[dict, int, int]:
    """Re-label *refl* into the indexing setting of *ref*.

    Picks the operator that maximises the hkl overlap with *ref*.
    :returns: ``(relabelled reflections, best overlap, second-best overlap)``
    """
    scored = []
    for op in ops:
        mapped = {op(*hkl): v for hkl, v in refl.items()}
        scored.append((len(ref.keys() & mapped.keys()), mapped))
    scored.sort(key=lambda x: x[0], reverse=True)
    second = scored[1][0] if len(scored) > 1 else 0
    return scored[0][1], scored[0][0], second


def decode_stream(
    stream_path: str,
    output_path: str,
    n_merged_frames: int = 3,
    split_file: Optional[str] = None,
    continuous_offset: Optional[int] = None,
    S_matrix: Optional[np.ndarray] = None,
    symmetry: Optional[str] = None,
    min_common: int = 1,
) -> Counter:
    """Decode a Hadamard-encoded CrystFEL stream into a stream of per-frame chunks.

    Chunks are gated into blocks either from *split_file* (see
    :func:`read_split_file`) or, when it is ``None``, from the encoder's naming
    convention ``{stem}_{tag}.h5`` where block = ``{stem}_{event}`` and the
    binary tag selects the pattern.

    A block is decoded only when all *n_merged_frames* patterns are present and
    each chunk holds exactly one crystal.  The n chunks are indexed independently,
    so with *symmetry* given each chunk is first re-labelled into the setting of
    the pattern-0 chunk using the Laue-group operator with the largest hkl
    overlap; blocks where that choice is not decisive (best <= 2x second best)
    are skipped.  Reflections are inner-joined on
    ``h k l``; intensities are inverted linearly and sigmas in quadrature.
    Peak, background and pixel columns are copied from the pattern-0 chunk.

    :param stream_path: Input stream (run ambigator on it first so the n chunks of a block share one indexing solution)
    :param output_path: Output stream path
    :param n_merged_frames: S-matrix order
    :param split_file: Optional gating file; see :func:`read_split_file`
    :param continuous_offset: Offset used during continuous encoding
    :param S_matrix: Pre-computed S-matrix; generated when ``None``
    :param symmetry: Laue group used to align indexing settings (``"6/m"``); ``None`` joins on raw hkl
    :param min_common: Skip blocks with fewer reflections common to all n chunks
    :returns: Counter of ``decoded``, ``unindexed``, ``incomplete``, ``multi_crystal``,
        ``ambiguous_setting``, ``too_few_common`` blocks, ``unmatched`` chunks and ``refl_decoded``
    """
    n = n_merged_frames
    S = generate_s_matrix(n) if S_matrix is None else S_matrix
    Sinv = compute_hadamard_inverse(S)
    tags = ["".join(str(int(x)) for x in row) for row in S]
    tag_index = {t: i for i, t in enumerate(tags)}
    tag_index.update({str(i): i for i in range(n)})

    split = read_split_file(split_file) if split_file else None
    tag_re = re.compile(r"^(.*)_([01]{%d})$" % n)

    def gate(chunk: dict) -> Optional[Tuple[str, int]]:
        if split is not None:
            hit = split.get((chunk["filename"], chunk["event"]))
            return (hit[0], tag_index[hit[1]]) if hit else None
        m = tag_re.match(Path(chunk["filename"]).stem)
        if not m or m.group(2) not in tag_index:
            return None
        return f"{m.group(1)}_{chunk['event']}", tag_index[m.group(2)]

    ops = _laue_ops(symmetry)
    blocks: Dict[str, Dict[int, dict]] = defaultdict(dict)
    order: List[str] = []
    stats: Counter = Counter()
    unindexed: set = set()
    header_parts: List[str] = []
    in_header = True
    with open(stream_path) as fh:
        buf: Optional[List[str]] = None
        for line in fh:
            if buf is None:
                if line.startswith(_CHUNK_BEGIN):
                    buf = []
                    in_header = False
                elif in_header:
                    header_parts.append(line)
                continue
            if line.startswith(_CHUNK_END):
                chunk = _parse_chunk("".join(buf))
                buf = None
                g = gate(chunk)
                if g is None:
                    stats["unmatched"] += 1
                    continue
                if g[0] not in blocks:
                    order.append(g[0])
                    blocks[g[0]] = {}
                if not chunk["reflections"]:
                    # unindexed: the block cannot be decoded, so keep no text
                    unindexed.add(g[0])
                    continue
                blocks[g[0]][g[1]] = chunk
                continue
            buf.append(line)
    header = "".join(header_parts)

    serial = 0
    with open(output_path, "w") as out:
        out.write(header)
        for block in order:
            chunks = blocks[block]
            if block in unindexed:
                stats["unindexed"] += 1
                continue
            if len(chunks) != n:
                stats["incomplete"] += 1
                continue
            # ponytail: one crystal per chunk, multi-lattice pairing needs orientation matching
            if any(len(c["reflections"]) != 1 for c in chunks.values()):
                stats["multi_crystal"] += 1
                continue
            refls = [chunks[i]["reflections"][0] for i in range(n)]
            if symmetry is not None:
                aligned, decisive = [refls[0]], True
                for r in refls[1:]:
                    a, best, second = _align_to(refls[0], r, ops)
                    decisive &= best > 2 * second
                    aligned.append(a)
                if not decisive:
                    stats["ambiguous_setting"] += 1
                    continue
                refls = aligned
            common = [hkl for hkl in refls[0] if all(hkl in r for r in refls[1:])]
            if len(common) < min_common:
                stats["too_few_common"] += 1
                continue
            stats["refl_decoded"] += len(common)
            I_enc = np.array([[r[hkl][0] for r in refls] for hkl in common]).reshape(-1, n)
            sig = np.array([[r[hkl][1] for r in refls] for hkl in common]).reshape(-1, n)
            I_dec, sig_dec = hadamard_decode(I_enc, sig, Sinv)
            for frame in range(n):
                idx = frame if continuous_offset is None else (frame - continuous_offset) % n
                lines = []
                for row, hkl in enumerate(common):
                    p = refls[0][hkl][2].split()
                    lines.append(
                        _REFL_FMT.format(
                            *hkl, I_dec[row, frame], sig_dec[row, frame],
                            float(p[5]), float(p[6]), float(p[7]), float(p[8]), p[9],
                        )
                    )
                serial += 1
                out.write(_CHUNK_BEGIN + "\n")
                out.write(_render_chunk(chunks[0]["text"], block, idx, serial, lines))
                out.write(_CHUNK_END + "\n")
            stats["decoded"] += 1

    print(f"Decoded {stats['decoded']} blocks -> {output_path}")
    for key in ("unindexed", "incomplete", "multi_crystal", "ambiguous_setting", "too_few_common", "unmatched"):
        if stats[key]:
            print(f"  skipped {stats[key]} {key.replace('_', ' ')}")
    return stats
