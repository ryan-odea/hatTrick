"""Tests for hatTrick.scale — post-decode resolution-shell scaling."""

import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hatTrick.scale.scaler import (
    compute_resolution,
    read_crystfel_hkl,
    scale_hkl_files,
    write_crystfel_hkl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CELL_CUBIC = (50.0, 50.0, 50.0, 90.0, 90.0, 90.0)
CELL_HEX = (78.0, 78.0, 37.0, 90.0, 90.0, 120.0)

SIMPLE_HKL = textwrap.dedent("""\
    CrystFEL reflection list version 2.0
    Symmetry: P6322
       h    k    l          I   sigma(I)
       1    0    0     100.0000      10.0000
       0    1    0     200.0000      20.0000
       1    1    0     150.0000      15.0000
       0    0    1     300.0000      30.0000
       1    0    1      80.0000       8.0000
       2    0    0      50.0000       5.0000
       0    2    0      60.0000       6.0000
       1    1    1     120.0000      12.0000
       2    1    0      90.0000       9.0000
       0    1    1     110.0000      11.0000
    End of reflections
""")

FULL_HKL = textwrap.dedent("""\
    CrystFEL reflection list version 2.0
    Symmetry: 6/m
       h    k    l          I    phase   sigma(I)   nmeas
       1    0    0     100.0000       0.0000      10.0000       5
       0    1    0     200.0000       0.0000      20.0000       3
    End of reflections
""")


def _write_hkl(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Round-trip I/O tests
# ---------------------------------------------------------------------------

class TestRoundTripIO:
    def test_simple_format_roundtrip(self, tmp_path):
        """read -> write -> read should preserve data and header."""
        src = _write_hkl(tmp_path, "orig.hkl", SIMPLE_HKL)
        df, header, sym = read_crystfel_hkl(str(src))

        assert sym == "P6322"
        assert df.height == 10
        assert set(df.columns) == {"h", "k", "l", "I", "sigma"}

        dst = tmp_path / "copy.hkl"
        write_crystfel_hkl(str(dst), df, header)

        df2, header2, sym2 = read_crystfel_hkl(str(dst))
        assert sym2 == "P6322"
        assert df2.height == df.height

        # Values should match
        for col in ["h", "k", "l"]:
            assert (df[col].to_numpy() == df2[col].to_numpy()).all()
        np.testing.assert_allclose(df["I"].to_numpy(), df2["I"].to_numpy(), atol=1e-3)
        np.testing.assert_allclose(df["sigma"].to_numpy(), df2["sigma"].to_numpy(), atol=1e-3)

    def test_full_format_roundtrip(self, tmp_path):
        """7-column CrystFEL format round-trips correctly."""
        src = _write_hkl(tmp_path, "full.hkl", FULL_HKL)
        df, header, sym = read_crystfel_hkl(str(src))

        assert sym == "6/m"
        assert df.height == 2
        assert "phase" in df.columns
        assert "nmeas" in df.columns

        dst = tmp_path / "copy_full.hkl"
        write_crystfel_hkl(str(dst), df, header)

        df2, _, sym2 = read_crystfel_hkl(str(dst))
        assert sym2 == "6/m"
        assert df2.height == 2
        np.testing.assert_allclose(df["I"].to_numpy(), df2["I"].to_numpy(), atol=1e-3)

    def test_header_footer_preserved(self, tmp_path):
        """The 'End of reflections' line should appear in output."""
        src = _write_hkl(tmp_path, "orig.hkl", SIMPLE_HKL)
        df, header, _ = read_crystfel_hkl(str(src))

        dst = tmp_path / "out.hkl"
        write_crystfel_hkl(str(dst), df, header)

        text = dst.read_text()
        assert "CrystFEL reflection list version 2.0" in text
        assert "End of reflections" in text


# ---------------------------------------------------------------------------
# Resolution computation
# ---------------------------------------------------------------------------

class TestComputeResolution:
    def test_cubic(self):
        """For cubic cell, 1/d^2 = (h^2+k^2+l^2)/a^2."""
        h = np.array([1, 0, 1, 2])
        k = np.array([0, 1, 1, 0])
        l = np.array([0, 0, 0, 0])
        inv_d_sq = compute_resolution(h, k, l, CELL_CUBIC)
        a = 50.0
        expected = (h**2 + k**2 + l**2) / a**2
        np.testing.assert_allclose(inv_d_sq, expected, rtol=1e-10)

    def test_hexagonal(self):
        """Verify hexagonal formula: 1/d^2 = (4/3)(h^2+hk+k^2)/a^2 + l^2/c^2."""
        a, c = 78.0, 37.0
        h = np.array([1, 0, 1, 0])
        k = np.array([0, 1, 1, 0])
        l = np.array([0, 0, 0, 1])
        inv_d_sq = compute_resolution(h, k, l, CELL_HEX)
        expected = (4.0 / 3.0) * (h**2 + h * k + k**2) / a**2 + l**2 / c**2
        np.testing.assert_allclose(inv_d_sq, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Scale recovery tests
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_refl: int, cell, scale_func, seed=42):
    """Generate synthetic dark + decoded data where decoded = dark / k(s).

    Returns (dark_df, dec_df, cell) where scale_func maps 1/d^2 -> k.
    """
    rng = np.random.default_rng(seed)
    # Random Miller indices
    h = rng.integers(-5, 6, n_refl)
    k = rng.integers(-5, 6, n_refl)
    l = rng.integers(-5, 6, n_refl)
    # Avoid (0,0,0)
    mask = (h != 0) | (k != 0) | (l != 0)
    h, k, l = h[mask], k[mask], l[mask]

    # Make unique
    hkl = np.column_stack([h, k, l])
    _, idx = np.unique(hkl, axis=0, return_index=True)
    h, k, l = h[idx], k[idx], l[idx]

    inv_d_sq = compute_resolution(h, k, l, cell)

    I_dark = rng.uniform(50, 500, len(h))
    sigma_dark = I_dark * 0.1
    k_true = scale_func(inv_d_sq)
    I_dec = I_dark / k_true
    sigma_dec = sigma_dark / np.abs(k_true)

    dark_df = pl.DataFrame({
        "h": h.astype(np.int32),
        "k": k.astype(np.int32),
        "l": l.astype(np.int32),
        "I": I_dark,
        "sigma": sigma_dark,
    })

    dec_df = pl.DataFrame({
        "h": h.astype(np.int32),
        "k": k.astype(np.int32),
        "l": l.astype(np.int32),
        "I": I_dec,
        "sigma": sigma_dec,
    })

    return dark_df, dec_df


class TestScaleRecovery:
    def test_constant_scale(self, tmp_path):
        """Recover a constant scale factor k=2."""
        dark_df, dec_df = _make_synthetic_data(200, CELL_HEX, lambda s: np.full_like(s, 2.0))

        # Write to files
        header = [
            "CrystFEL reflection list version 2.0\n",
            "Symmetry: P6322\n",
            "   h    k    l          I   sigma(I)\n",
            "End of reflections\n",
        ]
        dark_path = tmp_path / "dark.hkl"
        dec_path = tmp_path / "dec.hkl"
        write_crystfel_hkl(str(dark_path), dark_df, header)
        write_crystfel_hkl(str(dec_path), dec_df, header)

        results = scale_hkl_files(
            dark_path=str(dark_path),
            decoded_paths=[str(dec_path)],
            cell=CELL_HEX,
            spacegroup="P6322",
            n_shells=10,
            poly_order=1,
            output_dir=str(tmp_path / "out"),
        )

        # After scaling by k≈2, I_scaled ≈ I_dark
        scaled = results[0]
        # Re-read dark for comparison
        dark_re, _, _ = read_crystfel_hkl(str(dark_path))

        # Join on hkl to compare
        joined = dark_re.rename({"I": "I_dark"}).join(
            scaled.rename({"I": "I_scaled"}),
            on=["h", "k", "l"],
            how="inner",
        )
        ratio = joined["I_scaled"].to_numpy() / joined["I_dark"].to_numpy()
        np.testing.assert_allclose(ratio, 1.0, rtol=0.05)

    def test_linear_scale(self, tmp_path):
        """Recover a linear scale factor k(s) = 1.5 + 3*s."""
        def scale_fn(s):
            return 1.5 + 3.0 * s
        dark_df, dec_df = _make_synthetic_data(300, CELL_HEX, scale_fn)

        header = [
            "CrystFEL reflection list version 2.0\n",
            "Symmetry: P6322\n",
            "   h    k    l          I   sigma(I)\n",
            "End of reflections\n",
        ]
        dark_path = tmp_path / "dark.hkl"
        dec_path = tmp_path / "dec.hkl"
        write_crystfel_hkl(str(dark_path), dark_df, header)
        write_crystfel_hkl(str(dec_path), dec_df, header)

        results = scale_hkl_files(
            dark_path=str(dark_path),
            decoded_paths=[str(dec_path)],
            cell=CELL_HEX,
            spacegroup="P6322",
            n_shells=15,
            poly_order=3,
            output_dir=str(tmp_path / "out"),
        )

        scaled = results[0]
        dark_re, _, _ = read_crystfel_hkl(str(dark_path))

        joined = dark_re.rename({"I": "I_dark"}).join(
            scaled.rename({"I": "I_scaled"}),
            on=["h", "k", "l"],
            how="inner",
        )
        ratio = joined["I_scaled"].to_numpy() / joined["I_dark"].to_numpy()
        np.testing.assert_allclose(ratio, 1.0, rtol=0.1)

    def test_quadratic_scale(self, tmp_path):
        """Recover a quadratic scale factor k(s) = 1.0 + 2*s + 5*s^2."""
        def scale_fn(s):
            return 1.0 + 2.0 * s + 5.0 * s**2
        dark_df, dec_df = _make_synthetic_data(400, CELL_CUBIC, scale_fn)

        header = [
            "CrystFEL reflection list version 2.0\n",
            "Symmetry: P1\n",
            "   h    k    l          I   sigma(I)\n",
            "End of reflections\n",
        ]
        dark_path = tmp_path / "dark.hkl"
        dec_path = tmp_path / "dec.hkl"
        write_crystfel_hkl(str(dark_path), dark_df, header)
        write_crystfel_hkl(str(dec_path), dec_df, header)

        results = scale_hkl_files(
            dark_path=str(dark_path),
            decoded_paths=[str(dec_path)],
            cell=CELL_CUBIC,
            spacegroup="P1",
            n_shells=20,
            poly_order=3,
            output_dir=str(tmp_path / "out"),
        )

        scaled = results[0]
        dark_re, _, _ = read_crystfel_hkl(str(dark_path))

        joined = dark_re.rename({"I": "I_dark"}).join(
            scaled.rename({"I": "I_scaled"}),
            on=["h", "k", "l"],
            how="inner",
        )
        ratio = joined["I_scaled"].to_numpy() / joined["I_dark"].to_numpy()
        np.testing.assert_allclose(ratio, 1.0, rtol=0.15)


# ---------------------------------------------------------------------------
# Missing / partial overlap tests
# ---------------------------------------------------------------------------

class TestMissingReflections:
    def test_zero_overlap_raises(self, tmp_path):
        """No overlapping reflections should raise ValueError."""
        header = [
            "CrystFEL reflection list version 2.0\n",
            "Symmetry: P1\n",
            "   h    k    l          I   sigma(I)\n",
            "End of reflections\n",
        ]
        dark_df = pl.DataFrame({
            "h": [1, 2], "k": [0, 0], "l": [0, 0],
            "I": [100.0, 200.0], "sigma": [10.0, 20.0],
        }).cast({"h": pl.Int32, "k": pl.Int32, "l": pl.Int32})

        dec_df = pl.DataFrame({
            "h": [3, 4], "k": [0, 0], "l": [0, 0],
            "I": [50.0, 60.0], "sigma": [5.0, 6.0],
        }).cast({"h": pl.Int32, "k": pl.Int32, "l": pl.Int32})

        dark_path = tmp_path / "dark.hkl"
        dec_path = tmp_path / "dec.hkl"
        write_crystfel_hkl(str(dark_path), dark_df, header)
        write_crystfel_hkl(str(dec_path), dec_df, header)

        with pytest.raises(ValueError, match="No overlapping reflections"):
            scale_hkl_files(
                dark_path=str(dark_path),
                decoded_paths=[str(dec_path)],
                cell=CELL_CUBIC,
                n_shells=5,
                poly_order=1,
                output_dir=str(tmp_path / "out"),
            )

    def test_partial_overlap_scales_all(self, tmp_path):
        """Decoded reflections not in dark still get scaled via polynomial."""
        header = [
            "CrystFEL reflection list version 2.0\n",
            "Symmetry: P1\n",
            "   h    k    l          I   sigma(I)\n",
            "End of reflections\n",
        ]
        # Dark has reflections 1-5, decoded has 1-7
        rng = np.random.default_rng(99)
        n_dark = 50
        n_extra = 20
        h_common = rng.integers(1, 6, n_dark).astype(np.int32)
        k_common = rng.integers(0, 4, n_dark).astype(np.int32)
        l_common = rng.integers(0, 3, n_dark).astype(np.int32)

        # Make unique
        hkl = np.column_stack([h_common, k_common, l_common])
        _, idx = np.unique(hkl, axis=0, return_index=True)
        h_common = h_common[idx]
        k_common = k_common[idx]
        l_common = l_common[idx]

        I_dark = rng.uniform(100, 500, len(h_common))
        sigma_dark = I_dark * 0.1
        I_dec_common = I_dark / 2.0  # constant scale k=2
        sigma_dec_common = sigma_dark / 2.0

        # Extra decoded reflections
        h_extra = rng.integers(6, 10, n_extra).astype(np.int32)
        k_extra = rng.integers(0, 4, n_extra).astype(np.int32)
        l_extra = rng.integers(0, 3, n_extra).astype(np.int32)
        I_dec_extra = rng.uniform(50, 250, n_extra) / 2.0
        sigma_dec_extra = I_dec_extra * 0.1

        dark_df = pl.DataFrame({
            "h": h_common, "k": k_common, "l": l_common,
            "I": I_dark, "sigma": sigma_dark,
        })
        dec_df = pl.DataFrame({
            "h": np.concatenate([h_common, h_extra]),
            "k": np.concatenate([k_common, k_extra]),
            "l": np.concatenate([l_common, l_extra]),
            "I": np.concatenate([I_dec_common, I_dec_extra]),
            "sigma": np.concatenate([sigma_dec_common, sigma_dec_extra]),
        })

        dark_path = tmp_path / "dark.hkl"
        dec_path = tmp_path / "dec.hkl"
        write_crystfel_hkl(str(dark_path), dark_df, header)
        write_crystfel_hkl(str(dec_path), dec_df, header)

        results = scale_hkl_files(
            dark_path=str(dark_path),
            decoded_paths=[str(dec_path)],
            cell=CELL_CUBIC,
            n_shells=5,
            poly_order=1,
            output_dir=str(tmp_path / "out"),
        )

        scaled = results[0]
        # All decoded reflections should be in output (common + extra)
        assert scaled.height == dec_df.height
