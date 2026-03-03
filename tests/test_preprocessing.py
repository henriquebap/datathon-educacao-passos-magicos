"""Tests for src/preprocessing.py module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    drop_identifier_columns,
    handle_missing_values,
    encode_categorical_columns,
    normalize_features,
    extract_target,
    split_data,
    preprocess_pipeline,
)
from src.utils import PEDRA_MAP


class TestDropIdentifierColumns:
    """Tests for identifier column removal."""

    def test_drops_nome_column(self):
        df = pd.DataFrame({"NOME": ["Alice"], "INDE_2022": [7.5]})
        result = drop_identifier_columns(df)
        assert "NOME" not in result.columns
        assert "INDE_2022" in result.columns

    def test_drops_multiple_identifiers(self):
        df = pd.DataFrame({"NOME": ["A"], "ID": [1], "MATRICULA": ["M1"], "INDE_2022": [7.5]})
        result = drop_identifier_columns(df)
        assert "NOME" not in result.columns
        assert "ID" not in result.columns
        assert "MATRICULA" not in result.columns
        assert "INDE_2022" in result.columns

    def test_no_identifiers_returns_same(self):
        df = pd.DataFrame({"INDE_2022": [7.5], "IAA_2022": [6.0]})
        result = drop_identifier_columns(df)
        assert list(result.columns) == ["INDE_2022", "IAA_2022"]


class TestHandleMissingValues:
    """Tests for missing value handling."""

    def test_median_strategy(self):
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "B": [10.0, np.nan, 30.0, 40.0]})
        result = handle_missing_values(df, strategy="median")
        assert result.isnull().sum().sum() == 0
        assert result.loc[2, "A"] == 2.0  # median of [1, 2, 4]

    def test_mean_strategy(self):
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 3.0]})
        result = handle_missing_values(df, strategy="mean")
        assert result.isnull().sum().sum() == 0
        assert result.loc[2, "A"] == 2.0  # mean of [1, 2, 3]

    def test_drop_strategy(self):
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [10.0, 20.0, 30.0]})
        result = handle_missing_values(df, strategy="drop")
        assert len(result) == 2
        assert result.isnull().sum().sum() == 0

    def test_categorical_filled_with_mode(self):
        df = pd.DataFrame({"cat": ["A", "A", "B", None]})
        result = handle_missing_values(df)
        assert result.loc[3, "cat"] == "A"

    def test_no_missing_values_unchanged(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(result, df)


class TestEncodeCategoricalColumns:
    """Tests for categorical encoding."""

    def test_encodes_pedra_columns(self):
        df = pd.DataFrame({"PEDRA_2022": ["Quartzo", "Ágata", "Ametista", "Topázio"]})
        result, encoders = encode_categorical_columns(df)
        expected = [PEDRA_MAP["Quartzo"], PEDRA_MAP["Ágata"], PEDRA_MAP["Ametista"], PEDRA_MAP["Topázio"]]
        assert list(result["PEDRA_2022"]) == expected

    def test_encodes_other_categoricals(self):
        df = pd.DataFrame({"TURMA_2022": ["A", "B", "C", "A"]})
        result, encoders = encode_categorical_columns(df)
        assert result["TURMA_2022"].dtype in [np.int64, np.int32, int]

    def test_returns_encoders(self):
        df = pd.DataFrame({"TURMA_2022": ["A", "B", "C"]})
        result, encoders = encode_categorical_columns(df)
        assert "TURMA_2022" in encoders

    def test_no_categoricals_unchanged(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        result, encoders = encode_categorical_columns(df)
        assert len(encoders) == 0


class TestNormalizeFeatures:
    """Tests for feature normalization."""

    def test_normalizes_to_zero_mean(self):
        X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_test = pd.DataFrame({"A": [2.5, 3.5]})
        X_train_s, X_test_s, scaler = normalize_features(X_train, X_test)
        assert abs(X_train_s["A"].mean()) < 0.1

    def test_returns_scaler(self):
        X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"A": [2.0]})
        _, _, scaler = normalize_features(X_train, X_test)
        assert scaler is not None

    def test_shapes_preserved(self):
        X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"A": [2.0], "B": [5.0]})
        X_train_s, X_test_s, _ = normalize_features(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape


class TestExtractTarget:
    """Tests for target extraction."""

    def test_extracts_defasagem_column(self):
        df = pd.DataFrame({
            "INDE_2022": [7.5, 6.0],
            "DEFASAGEM_2022": [0, 1],
        })
        X, y = extract_target(df, target_year=2022)
        assert "DEFASAGEM_2022" not in X.columns
        assert len(y) == 2
        assert set(y.unique()).issubset({0, 1})

    def test_uses_latest_year_by_default(self, synthetic_data):
        X, y = extract_target(synthetic_data)
        # y may be smaller than original if NaN target rows are dropped
        assert len(y) <= len(synthetic_data)
        assert len(y) > 0
        assert len(X) == len(y)

    def test_target_is_binary(self, synthetic_data):
        X, y = extract_target(synthetic_data)
        assert set(y.unique()).issubset({0, 1})

    def test_raises_on_missing_target(self):
        df = pd.DataFrame({"INDE_2022": [7.5]})
        with pytest.raises(ValueError, match="Coluna target"):
            extract_target(df, target_year=2022)


class TestSplitData:
    """Tests for data splitting."""

    def test_correct_proportions(self):
        X = pd.DataFrame({"A": range(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_stratified_split(self):
        X = pd.DataFrame({"A": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        # Check rough stratification
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.1

    def test_reproducible_with_seed(self):
        X = pd.DataFrame({"A": range(50)})
        y = pd.Series([0] * 25 + [1] * 25)
        X1, _, _, _ = split_data(X, y, random_state=42)
        X2, _, _, _ = split_data(X, y, random_state=42)
        pd.testing.assert_frame_equal(X1, X2)


class TestPreprocessPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_pipeline_returns_expected_outputs(self, synthetic_data):
        X_train, X_test, y_train, y_test, artifacts = preprocess_pipeline(synthetic_data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(artifacts, dict)

    def test_pipeline_produces_no_nulls(self, synthetic_data):
        X_train, X_test, _, _, _ = preprocess_pipeline(synthetic_data)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_pipeline_artifacts_contain_keys(self, synthetic_data):
        _, _, _, _, artifacts = preprocess_pipeline(synthetic_data)
        assert "scaler" in artifacts
        assert "encoders" in artifacts
        assert "feature_names" in artifacts

    def test_pipeline_features_match(self, synthetic_data):
        X_train, X_test, _, _, artifacts = preprocess_pipeline(synthetic_data)
        assert list(X_train.columns) == list(X_test.columns)
        assert artifacts["feature_names"] == list(X_train.columns)
