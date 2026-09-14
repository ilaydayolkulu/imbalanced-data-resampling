"""SMOTE (Synthetic Minority Over-sampling Technique) resampler implementation."""

from typing import Tuple, Union

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.exceptions import ResamplingExecutionError
from src.resamplers.base import BaseResampler


class SMOTEResampler(BaseResampler):
    """Generates synthetic minority instances via linear interpolation along k-NN vectors."""

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: Union[str, float] = "auto",
        random_state: int = 42,
    ):
        """Initializes SMOTE with k-NN parameter, sampling ratio/strategy, and random seed.

        Args:
            k_neighbors: Number of nearest neighbors to construct synthetic samples.
            sampling_strategy: Sampling ratio (float) or predefined strategy ('auto', 'minority').
            random_state: Random seed for reproducible neighbor sampling.
        """
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "SMOTE"

    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Executes SMOTE over-sampling on feature matrix X and target labels y.

        Args:
            X: Input numerical feature DataFrame.
            y: Binary or multi-class target Series.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled feature DataFrame and target Series.
        """
        try:
            smote = SMOTE(
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
            X_res, y_res = smote.fit_resample(X, y)

            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X.columns)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y.name)

            self.last_execution_stats = {
                "synthetic_generated": len(X_res) - len(X),
            }

            return X_res, y_res
        except Exception as exc:
            raise ResamplingExecutionError(f"SMOTE execution failed: {str(exc)}") from exc
