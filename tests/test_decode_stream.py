from pathlib import Path

import numpy as np

from hatTrick._helpers import compute_hadamard_inverse, generate_s_matrix, hadamard_decode
from hatTrick.decode.stream import _parse_chunk, decode_stream

HKL = [(0, 0, 2), (1, 0, 0), (1, 1, 0), (2, 0, 1)]


def _chunk(fname, event, refl, crystals=1):
    body = f"----- Begin chunk -----\nImage filename: {fname}\nEvent: //{event}\nImage serial number: 1\nhit = 1\n"
    for _ in range(crystals):
        body += "--- Begin crystal\nCell parameters 7.59 7.59 3.78 nm, 90 90 120 deg\n"
        body += f"num_reflections = {len(refl)}\nReflections measured after indexing\n"
        body += "   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n"
        for (h, k, l), (I, s) in refl.items():
            body += f"{h:4d} {k:4d} {l:4d} {I:10.2f} {s:10.2f} {I:10.2f} {1.0:10.2f} {10.0:6.1f} {20.0:6.1f} p0\n"
        body += "End of reflections\n--- End crystal\n"
    return body + "----- End chunk -----\n"


def _make_stream(tmp_path, n=3, n_blocks=2, naming=True, seed=0):
    rng = np.random.default_rng(seed)
    S = generate_s_matrix(n)
    tags = ["".join(str(x) for x in row) for row in S]
    truth = rng.uniform(10, 100, (n_blocks, n, len(HKL)))
    sig = rng.uniform(1, 5, (n_blocks, n, len(HKL)))
    stream = "CrystFEL stream format 2.3\n----- Begin geometry file -----\n----- End geometry file -----\n"
    split_lines = []
    for b in range(n_blocks):
        for p in range(n):
            I_enc = S[p] @ truth[b]
            s_enc = S[p] @ sig[b]
            fname = f"run_{tags[p]}.h5" if naming else f"weird{p}.h5"
            event = b if naming else b * 7 + p
            stream += _chunk(fname, event, dict(zip(HKL, zip(I_enc, s_enc))))
            split_lines.append(f"{fname} {event} blk{b} {p}")
    (tmp_path / "in.stream").write_text(stream)
    (tmp_path / "split.txt").write_text("\n".join(split_lines) + "\n")
    return truth, S @ sig, compute_hadamard_inverse(S)  # sig -> encoded sigmas per block


def _read_decoded(path):
    out = {}
    for raw in Path(path).read_text().split("----- Begin chunk -----\n")[1:]:
        c = _parse_chunk(raw)
        out[(c["filename"], int(c["event"]))] = c["reflections"][0]
    return out


def test_hadamard_decode_sigma_quadrature():
    S = generate_s_matrix(3)
    Sinv = compute_hadamard_inverse(S)
    sig = np.array([[1.0, 1.0, 10.0]])  # linear propagation would go negative here
    I_dec, s_dec = hadamard_decode(np.ones((1, 3)), sig, Sinv)
    assert np.all(s_dec > 0)
    np.testing.assert_allclose(s_dec, 0.5 * np.sqrt(102.0))


def test_decode_stream_default_naming(tmp_path):
    truth, sig, Sinv = _make_stream(tmp_path)
    stats = decode_stream(str(tmp_path / "in.stream"), str(tmp_path / "out.stream"), 3)
    assert stats["decoded"] == 2
    dec = _read_decoded(tmp_path / "out.stream")
    for b in range(2):
        for f in range(3):
            refl = dec[(f"run_{b}", f)]
            I = np.array([refl[hkl][0] for hkl in HKL])
            s = np.array([refl[hkl][1] for hkl in HKL])
            np.testing.assert_allclose(I, truth[b, f], atol=0.01)
            np.testing.assert_allclose(s, np.sqrt((Sinv[f] ** 2) @ sig[b] ** 2), atol=0.01)  # quadrature


def test_decode_stream_split_file(tmp_path):
    truth, _, _ = _make_stream(tmp_path, naming=False)
    stats = decode_stream(str(tmp_path / "in.stream"), str(tmp_path / "out.stream"), 3)
    assert stats["decoded"] == 0 and stats["unmatched"] == 6
    stats = decode_stream(
        str(tmp_path / "in.stream"), str(tmp_path / "out.stream"), 3, split_file=str(tmp_path / "split.txt")
    )
    assert stats["decoded"] == 2
    dec = _read_decoded(tmp_path / "out.stream")
    I = np.array([dec[("blk1", 2)][hkl][0] for hkl in HKL])
    np.testing.assert_allclose(I, truth[1, 2], atol=0.01)


def test_decode_stream_skips_incomplete_and_multi(tmp_path):
    _make_stream(tmp_path, n_blocks=1)
    s = (tmp_path / "in.stream").read_text()
    chunks = s.split("----- Begin chunk -----\n")
    extra = _chunk("run_110.h5", 5, {HKL[0]: (1.0, 1.0)}, crystals=2)  # lone block, 2 crystals
    (tmp_path / "in.stream").write_text(s + extra)
    stats = decode_stream(str(tmp_path / "in.stream"), str(tmp_path / "out.stream"), 3)
    assert stats["decoded"] == 1 and stats["incomplete"] == 1
    (tmp_path / "in.stream").write_text(chunks[0] + "----- Begin chunk -----\n".join([""] + chunks[1:]).replace(
        "--- End crystal\n", "--- End crystal\n--- Begin crystal\nnum_reflections = 0\nReflections measured after indexing\nhdr\nEnd of reflections\n--- End crystal\n", 1))
    stats = decode_stream(str(tmp_path / "in.stream"), str(tmp_path / "out.stream"), 3)
    assert stats["multi_crystal"] == 1
