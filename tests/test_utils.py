"""Tests for src/utils.py module."""

import pandas as pd
import numpy as np
import pytest

from src.utils import (
    generate_synthetic_data,
    get_latest_year,
    get_year_columns,
    NUMERIC_INDICATORS,
    YEARS,
    PEDRA_MAP,
)


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_generates_correct_number_of_samples(self):
        df = generate_synthetic_data(n_samples=100)
        assert len(df) == 100

    def test_generates_different_sample_counts(self):
        df = generate_synthetic_data(n_samples=50)
        assert len(df) == 50

    def test_contains_year_suffixed_columns(self):
        df = generate_synthetic_data(n_samples=10)
        for year in YEARS:
            assert f"INDE_{year}" in df.columns
            assert f"IAA_{year}" in df.columns

    def test_contains_defasagem_target(self):
        df = generate_synthetic_data(n_samples=10)
        for year in YEARS:
            assert f"DEFASAGEM_{year}" in df.columns

    def test_defasagem_is_binary(self):
        df = generate_synthetic_data(n_samples=100)
        for year in YEARS:
            col = f"DEFASAGEM_{year}"
            unique_vals = df[col].dropna().unique()
            assert set(unique_vals).issubset({0, 1, 0.0, 1.0})

    def test_pedra_values_are_valid(self):
        df = generate_synthetic_data(n_samples=100)
        valid_pedras = set(PEDRA_MAP.keys())
        for year in YEARS:
            col = f"PEDRA_{year}"
            actual = set(df[col].dropna().unique())
            assert actual.issubset(valid_pedras)

    def test_contains_missing_values(self):
        df = generate_synthetic_data(n_samples=200, seed=42)
        assert df.isnull().sum().sum() > 0

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_data(n_samples=50, seed=123)
        df2 = generate_synthetic_data(n_samples=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_data(n_samples=50, seed=1)
        df2 = generate_synthetic_data(n_samples=50, seed=2)
        assert not df1.equals(df2)

    def test_numeric_indicators_in_range(self):
        df = generate_synthetic_data(n_samples=100)
        for year in YEARS:
            for indicator in NUMERIC_INDICATORS:
                col = f"{indicator}_{year}"
                if col in df.columns:
                    values = df[col].dropna()
                    assert values.min() >= 0
                    assert values.max() <= 10


class TestGetLatestYear:
    """Tests for year detection."""

    def test_detects_latest_year(self):
        df = pd.DataFrame({"INDE_2020": [1], "INDE_2021": [2], "INDE_2022": [3]})
        assert get_latest_year(df) == 2022

    def test_single_year(self):
        df = pd.DataFrame({"INDE_2022": [1], "IAA_2022": [2]})
        assert get_latest_year(df) == 2022

    def test_raises_on_no_year_columns(self):
        df = pd.DataFrame({"name": ["test"], "value": [1]})
        with pytest.raises(ValueError, match="coluna com sufixo de ano"):
            get_latest_year(df)


class TestGetYearColumns:
    """Tests for year column filtering."""

    def test_returns_columns_for_year(self):
        df = pd.DataFrame({
            "INDE_2020": [1], "IAA_2020": [2],
            "INDE_2021": [3], "IAA_2021": [4],
        })
        cols = get_year_columns(df, 2020)
        assert "INDE_2020" in cols
        assert "IAA_2020" in cols
        assert "INDE_2021" not in cols

    def test_empty_for_nonexistent_year(self):
        df = pd.DataFrame({"INDE_2020": [1]})
        cols = get_year_columns(df, 2025)
        assert cols == []
