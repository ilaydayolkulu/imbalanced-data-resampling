"""ADASYN (Adaptive Synthetic Sampling) resampler implementation."""

from typing import Tuple, Union

import pandas as pd
from imblearn.over_sampling import ADASYN

from src.exceptions import ResamplingExecutionError
from src.resamplers.base import BaseResampler


class ADASYNResampler(BaseResampler):
    """Generates synthetic minority instances adaptively based on local border difficulty."""

    def __init__(
        self,
        n_neighbors: int = 5,
        sampling_strategy: Union[str, float] = "auto",
        random_state: int = 42,
    ):
        """Initializes ADASYN with n-neighbors, sampling strategy, and random seed.

        Args:
            n_neighbors: Number of nearest neighbors used to compute the difficulty ratio.
            sampling_strategy: Sampling ratio (float) or predefined strategy ('auto', 'minority').
            random_state: Random seed for reproducible generation.
        """
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "ADASYN"

    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Executes ADASYN over-sampling with defensive error handling for zero-density regions.

        Args:
            X: Input numerical feature DataFrame.
            y: Binary or multi-class target Series.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled feature DataFrame and target Series.
        """
        try:
            adasyn = ADASYN(
                n_neighbors=self.n_neighbors,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
            X_res, y_res = adasyn.fit_resample(X, y)

            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X.columns)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y.name)

            return X_res, y_res
        except RuntimeError as rerr:
            raise ResamplingExecutionError(
                f"ADASYN failed to generate synthetic samples: {str(rerr)}. "
                "This typically occurs when minority samples have no majority neighbors within the "
                f"specified neighborhood (n_neighbors={self.n_neighbors}). Consider switching to SMOTE or "
                "increasing n_neighbors."
            ) from rerr
        except Exception as exc:
            raise ResamplingExecutionError(f"ADASYN execution failed: {str(exc)}") from exc
