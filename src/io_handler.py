"""Data input/output management, schema verification, and tabular data validation."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.exceptions import DataValidationError, FileIOError


class IOHandler:
    """Handles reading, validating, and writing imbalanced tabular datasets."""

    @staticmethod
    def load_dataset(file_path: Path) -> pd.DataFrame:
        """Loads a CSV file into a pandas DataFrame with validation.

        Args:
            file_path: Absolute or relative path to CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileIOError: If file does not exist or cannot be read.
        """
        if not file_path.exists():
            raise FileIOError(f"Input file not found at: {file_path}")

        if not file_path.is_file():
            raise FileIOError(f"Specified path is not a valid file: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            raise FileIOError(f"Failed to read CSV from '{file_path}': {str(exc)}") from exc

        if df.empty:
            raise DataValidationError(f"Input dataset '{file_path}' is empty.")

        return df

    @staticmethod
    def validate_and_split(
        df: pd.DataFrame,
        target_col: str,
        k_neighbors: int,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[int, int]]:
        """Validates feature matrices, target distribution, and k-NN feasibility.

        Args:
            df: Raw input DataFrame.
            target_col: Name of the target label / class column.
            k_neighbors: Required nearest neighbors for the resampling algorithm.

        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict[int, int]]:
                - X: Feature matrix.
                - y: Target series.
                - class_counts: Class distribution dictionary before resampling.

        Raises:
            DataValidationError: If target column is missing, features contain NaNs,
                                 or minority class has fewer samples than k_neighbors.
        """
        if target_col not in df.columns:
            available = ", ".join(list(df.columns[:10]))
            raise DataValidationError(
                f"Target column '{target_col}' not found in dataset. "
                f"Available columns: [{available}{'...' if len(df.columns) > 10 else ''}]"
            )

        # Check for missing values (NaNs)
        nan_counts = df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            bad_cols = [f"{col} ({count} NaNs)" for col, count in cols_with_nan.items()]
            raise DataValidationError(
                f"Dataset contains missing values (NaNs). "
                f"Resampling requires clean numerical data. Columns with NaNs: {', '.join(bad_cols)}"
            )

        # Separate features and target
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Verify class distribution
        class_counts = y.value_counts().to_dict()
        if len(class_counts) < 2:
            raise DataValidationError(
                f"Target column '{target_col}' must contain at least 2 distinct classes. "
                f"Found only {len(class_counts)} class: {class_counts}"
            )

        # Check minority sample count against k_neighbors
        min_class_samples = min(class_counts.values())
        if min_class_samples <= k_neighbors:
            raise DataValidationError(
                f"Insufficient minority class samples for k-NN graph. "
                f"Minority class has only {min_class_samples} samples, but k_neighbors is {k_neighbors}. "
                f"Please set --k_neighbors to {max(1, min_class_samples - 1)} or lower."
            )

        # Ensure all feature columns are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            raise DataValidationError(
                f"Resampling algorithms require numeric features. "
                f"Found non-numeric columns: {non_numeric_cols}. "
                f"Please encode or drop them before running resampling."
            )

        return X, y, class_counts

    @staticmethod
    def save_dataset(X: pd.DataFrame, y: pd.Series, target_col: str, output_path: Path) -> None:
        """Merges balanced features and target, then writes to disk.

        Args:
            X: Resampled feature DataFrame.
            y: Resampled target Series.
            target_col: Name of the target column.
            output_path: Destination path for the balanced CSV.

        Raises:
            FileIOError: If directory cannot be created or file cannot be written.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resampled_df = X.copy()
            resampled_df[target_col] = y.values
            resampled_df.to_csv(output_path, index=False)
        except Exception as exc:
            raise FileIOError(f"Failed to save resampled CSV to '{output_path}': {str(exc)}") from exc
