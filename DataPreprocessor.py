from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-safe preprocessing for cross-sectional pandas data.

    Parameters
    ----------
    categorical_cols:
        Nominal categorical columns to one-hot encode.

    ordinal_categories:
        Mapping from each ordinal column to its ordered categories.

        Example:
        {
            "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        }

    numerical_cols:
        Numerical columns to standardize.

    remainder:
        Determines what happens to columns not assigned to a transformation:
        - "drop": remove them
        - "passthrough": leave them unchanged

    sparse_output:
        If True, return a scipy sparse matrix.
        If False, return a pandas DataFrame.

    handle_unknown_ordinal:
        Numeric code assigned to unseen ordinal categories.
    """

    def __init__(
        self,
        categorical_cols: Sequence[str] | None = None,
        ordinal_categories: Mapping[str, Sequence[Any]] | None = None,
        numerical_cols: Sequence[str] | None = None,
        *,
        remainder: Literal["drop", "passthrough"] = "drop",
        sparse_output: bool = False,
        handle_unknown_ordinal: int = -1,
    ) -> None:
        # sklearn convention:
        # __init__ should only store user-provided parameters.
        self.categorical_cols = categorical_cols
        self.ordinal_categories = ordinal_categories
        self.numerical_cols = numerical_cols
        self.remainder = remainder
        self.sparse_output = sparse_output
        self.handle_unknown_ordinal = handle_unknown_ordinal

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "DataPreprocessor":
        """
        Learn category mappings and scaling statistics from training data.

        This method should only be called on training data to avoid leakage.
        """
        X = self._validate_X(X)

        categorical_cols = list(self.categorical_cols or [])
        ordinal_categories = dict(self.ordinal_categories or {})
        ordinal_cols = list(ordinal_categories)
        numerical_cols = list(self.numerical_cols or [])

        self._validate_configuration(
            X,
            categorical_cols=categorical_cols,
            ordinal_cols=ordinal_cols,
            numerical_cols=numerical_cols,
            ordinal_categories=ordinal_categories,
        )

        transformers: list[tuple[str, Any, list[str]]] = []

        if categorical_cols:
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=self.sparse_output,
            )

            transformers.append(
                (
                    "categorical",
                    categorical_transformer,
                    categorical_cols,
                )
            )

        if ordinal_cols:
            ordered_categories = [
                list(ordinal_categories[column])
                for column in ordinal_cols
            ]

            ordinal_transformer = OrdinalEncoder(
                categories=ordered_categories,
                handle_unknown="use_encoded_value",
                unknown_value=self.handle_unknown_ordinal,
            )

            transformers.append(
                (
                    "ordinal",
                    ordinal_transformer,
                    ordinal_cols,
                )
            )

        if numerical_cols:
            numerical_transformer = StandardScaler()

            transformers.append(
                (
                    "numerical",
                    numerical_transformer,
                    numerical_cols,
                )
            )

        if not transformers and self.remainder == "drop":
            raise ValueError(
                "No transformations were configured. Provide categorical_cols, "
                "ordinal_categories, numerical_cols, or use "
                "remainder='passthrough'."
            )

        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder=self.remainder,
            verbose_feature_names_out=False,
            sparse_threshold=1.0 if self.sparse_output else 0.0,
        )

        self.column_transformer_.fit(X, y)

        # Learned attributes end in an underscore by sklearn convention.
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply preprocessing learned during fit().

        fit() must be called before transform().
        """
        check_is_fitted(self, "column_transformer_")

        X = self._validate_X(X)

        expected_columns = list(self.feature_names_in_)

        missing_columns = sorted(
            set(expected_columns) - set(X.columns)
        )

        if missing_columns:
            raise ValueError(
                "X is missing columns that were present during fit: "
                f"{missing_columns}"
            )

        # Ensure columns appear in the same order as during fitting.
        # Extra columns are ignored.
        X_ordered = X.loc[:, expected_columns]

        transformed = self.column_transformer_.transform(X_ordered)

        if self.sparse_output:
            return transformed

        return pd.DataFrame(
            transformed,
            columns=self.get_feature_names_out(),
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        """
        Return the names of the processed output features.
        """
        check_is_fitted(self, "column_transformer_")

        return self.column_transformer_.get_feature_names_out(
            input_features
        )

    def split_fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        test_size: float = 0.2,
        random_state: int | None = None,
        stratify: pd.Series | None = None,
        shuffle: bool = True,
    ):
        """
        Split raw data, fit on training data, and transform both partitions.

        This method avoids leakage because the encoders and scaler are fitted
        only on X_train.

        Returns
        -------
        X_train_processed
        X_test_processed
        y_train
        y_test
        """
        X, y = self._validate_supervised_inputs(X, y)

        self._validate_split_arguments(
            test_size=test_size,
            stratify=stratify,
            shuffle=shuffle,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )

        X_train_processed = self.fit_transform(
            X_train,
            y_train,
        )

        X_test_processed = self.transform(X_test)

        return (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
        )

    @staticmethod
    def split(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        test_size: float = 0.2,
        random_state: int | None = None,
        stratify: pd.Series | None = None,
        shuffle: bool = True,
    ):
        """
        Split raw X and y without fitting any preprocessing.
        """
        X, y = DataPreprocessor._validate_supervised_inputs(X, y)

        DataPreprocessor._validate_split_arguments(
            test_size=test_size,
            stratify=stratify,
            shuffle=shuffle,
        )

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the feature DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame; "
                f"got {type(X).__name__}."
            )

        if X.empty:
            raise ValueError("X cannot be empty.")

        if X.columns.duplicated().any():
            duplicate_columns = (
                X.columns[X.columns.duplicated()].tolist()
            )

            raise ValueError(
                "X contains duplicate column names: "
                f"{duplicate_columns}"
            )

        return X

    @staticmethod
    def _validate_supervised_inputs(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Validate X and y for supervised learning.
        """
        X = DataPreprocessor._validate_X(X)

        if not isinstance(y, pd.Series):
            raise TypeError(
                "y must be a pandas Series; "
                f"got {type(y).__name__}."
            )

        if y.empty:
            raise ValueError("y cannot be empty.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must have equal length; "
                f"got {len(X)} and {len(y)}."
            )

        if not X.index.equals(y.index):
            raise ValueError(
                "X and y must have identical indices and row ordering."
            )

        return X, y

    @staticmethod
    def _validate_split_arguments(
        test_size: float,
        stratify: pd.Series | None,
        shuffle: bool,
    ) -> None:
        """
        Validate train-test split arguments.
        """
        if (
            not isinstance(test_size, (int, float))
            or isinstance(test_size, bool)
        ):
            raise TypeError(
                "test_size must be a float between 0 and 1."
            )

        if not 0 < float(test_size) < 1:
            raise ValueError(
                "test_size must be between 0 and 1; "
                f"got {test_size}."
            )

        if stratify is not None and not shuffle:
            raise ValueError(
                "stratify cannot be used when shuffle=False."
            )

    def _validate_configuration(
        self,
        X: pd.DataFrame,
        *,
        categorical_cols: list[str],
        ordinal_cols: list[str],
        numerical_cols: list[str],
        ordinal_categories: dict[str, Sequence[Any]],
    ) -> None:
        """
        Validate the requested preprocessing configuration.
        """
        if self.remainder not in {"drop", "passthrough"}:
            raise ValueError(
                "remainder must be 'drop' or 'passthrough'."
            )

        if not isinstance(self.sparse_output, bool):
            raise TypeError("sparse_output must be a bool.")

        if not isinstance(self.handle_unknown_ordinal, int):
            raise TypeError(
                "handle_unknown_ordinal must be an int."
            )

        configured_columns = (
            categorical_cols
            + ordinal_cols
            + numerical_cols
        )

        non_string_columns = [
            column
            for column in configured_columns
            if not isinstance(column, str)
        ]

        if non_string_columns:
            raise TypeError(
                "All configured column names must be strings: "
                f"{non_string_columns}"
            )

        missing_columns = sorted(
            set(configured_columns) - set(X.columns)
        )

        if missing_columns:
            raise ValueError(
                "Configured columns are absent from X: "
                f"{missing_columns}"
            )

        seen: set[str] = set()
        overlapping_columns: set[str] = set()

        for column in configured_columns:
            if column in seen:
                overlapping_columns.add(column)

            seen.add(column)

        if overlapping_columns:
            raise ValueError(
                "Each column may belong to only one transformation group; "
                f"duplicates: {sorted(overlapping_columns)}"
            )

        for column, categories in ordinal_categories.items():
            category_values = list(categories)

            if not category_values:
                raise ValueError(
                    f"Ordinal category order for '{column}' "
                    "cannot be empty."
                )

            try:
                number_unique = len(set(category_values))
            except TypeError as error:
                raise TypeError(
                    f"Ordinal categories for '{column}' must "
                    "contain hashable values."
                ) from error

            if len(category_values) != number_unique:
                raise ValueError(
                    f"Ordinal category order for '{column}' "
                    "contains duplicates."
                )

            learned_codes = range(len(category_values))

            if self.handle_unknown_ordinal in learned_codes:
                raise ValueError(
                    "handle_unknown_ordinal="
                    f"{self.handle_unknown_ordinal} conflicts with "
                    f"learned codes 0 through "
                    f"{len(category_values) - 1} for '{column}'."
                )