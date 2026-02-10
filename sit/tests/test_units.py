"""Tests for sit.core.units module.

Covers:
- Roundtrip conversions between all unit pairs
- Canonical constant correctness
- Invalid unit rejection
- Vectorized (numpy array) conversions
"""

import numpy as np
import pytest

from sit.core.units import (
    CANONICAL_LATENCY_UNIT,
    LatencyUnit,
    assert_latency_unit,
    convert_latency,
    validate_latency_unit,
)


class TestCanonicalUnit:
    def test_canonical_is_us(self):
        assert CANONICAL_LATENCY_UNIT == "us"

    def test_assert_canonical_passes(self):
        assert_latency_unit("us")  # Should not raise

    def test_assert_non_canonical_fails(self):
        with pytest.raises(ValueError, match="Expected canonical"):
            assert_latency_unit("ns")

        with pytest.raises(ValueError, match="Expected canonical"):
            assert_latency_unit("ms")


class TestConvertLatency:
    def test_identity(self):
        assert convert_latency(100.0, "us", "us") == 100.0
        assert convert_latency(100.0, "ns", "ns") == 100.0
        assert convert_latency(100.0, "ms", "ms") == 100.0

    def test_us_to_ns(self):
        result = convert_latency(1.0, "us", "ns")
        assert result == pytest.approx(1000.0)

    def test_ns_to_us(self):
        result = convert_latency(1000.0, "ns", "us")
        assert result == pytest.approx(1.0)

    def test_us_to_ms(self):
        result = convert_latency(1000.0, "us", "ms")
        assert result == pytest.approx(1.0)

    def test_ms_to_us(self):
        result = convert_latency(1.0, "ms", "us")
        assert result == pytest.approx(1000.0)

    def test_ns_to_ms(self):
        result = convert_latency(1_000_000.0, "ns", "ms")
        assert result == pytest.approx(1.0)

    def test_ms_to_ns(self):
        result = convert_latency(1.0, "ms", "ns")
        assert result == pytest.approx(1_000_000.0)

    def test_roundtrip_us_ns(self):
        original = 42.5
        converted = convert_latency(original, "us", "ns")
        back = convert_latency(converted, "ns", "us")
        assert back == pytest.approx(original)

    def test_roundtrip_us_ms(self):
        original = 42.5
        converted = convert_latency(original, "us", "ms")
        back = convert_latency(converted, "ms", "us")
        assert back == pytest.approx(original)

    def test_roundtrip_ns_ms(self):
        original = 42.5
        converted = convert_latency(original, "ns", "ms")
        back = convert_latency(converted, "ms", "ns")
        assert back == pytest.approx(original)

    def test_vectorized_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = convert_latency(arr, "us", "ns")
        expected = np.array([1000.0, 2000.0, 3000.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_vectorized_roundtrip(self):
        arr = np.array([10.0, 20.0, 30.0])
        converted = convert_latency(arr, "ms", "ns")
        back = convert_latency(converted, "ns", "ms")
        np.testing.assert_array_almost_equal(back, arr)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            convert_latency(1.0, "invalid", "us")

        with pytest.raises(ValueError):
            convert_latency(1.0, "us", "seconds")

    def test_enum_input(self):
        result = convert_latency(1.0, LatencyUnit.US, LatencyUnit.NS)
        assert result == pytest.approx(1000.0)


class TestValidateLatencyUnit:
    def test_valid_units(self):
        assert validate_latency_unit("us") == LatencyUnit.US
        assert validate_latency_unit("ns") == LatencyUnit.NS
        assert validate_latency_unit("ms") == LatencyUnit.MS

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid latency unit"):
            validate_latency_unit("seconds")
