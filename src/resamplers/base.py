"""Abstract base interface for all resampling strategies."""

from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd


class BaseResampler(ABC):
    """Abstract Strategy interface for imbalanced data resampling."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the human-readable name of the resampling algorithm."""
        pass

    @abstractmethod
    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Performs synthetic resampling on feature matrix X and target y.

        Args:
            X: Input numerical feature DataFrame.
            y: Target binary/multi-class label Series.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled feature DataFrame and target Series.

        Raises:
            ResamplingExecutionError: If the resampling algorithm encounters an execution error.
        """
        pass
