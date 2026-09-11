"""SMOTE-TOMEK hybrid resampler implementation combining SMOTE and Tomek Links."""

from typing import Tuple, Union

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from src.exceptions import ResamplingExecutionError
from src.resamplers.base import BaseResampler


class SMOTETomekResampler(BaseResampler):
    """Hybrid resampler that over-samples minority instances via SMOTE and cleans boundaries via Tomek Links."""

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: Union[str, float] = "auto",
        random_state: int = 42,
        tomek_sampling_strategy: str = "auto",
    ):
        """Initializes SMOTE-TOMEK with inner SMOTE and Tomek Links parameters.

        Args:
            k_neighbors: Number of nearest neighbors for inner SMOTE over-sampling.
            sampling_strategy: Sampling strategy/ratio for inner SMOTE.
            random_state: Seed for random number generators.
            tomek_sampling_strategy: Tomek links sample removal policy ('auto', 'all', or 'not minority').
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.tomek_sampling_strategy = tomek_sampling_strategy

    @property
    def name(self) -> str:
        return "SMOTE-TOMEK"

    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Executes two-stage SMOTE-TOMEK pipeline on feature matrix X and target labels y.

        Args:
            X: Input numerical feature DataFrame.
            y: Binary or multi-class target Series.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled feature DataFrame and target Series.
        """
        try:
            inner_smote = SMOTE(
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

            inner_tomek = TomekLinks(
                sampling_strategy=self.tomek_sampling_strategy,
            )

            resampler = SMOTETomek(
                smote=inner_smote,
                tomek=inner_tomek,
                random_state=self.random_state,
            )

            X_res, y_res = resampler.fit_resample(X, y)

            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X.columns)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y.name)

            return X_res, y_res
        except Exception as exc:
            raise ResamplingExecutionError(f"SMOTE-TOMEK execution failed: {str(exc)}") from exc
