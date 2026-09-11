"""Factory for instantiating concrete BaseResampler instances based on configuration."""

from src.config import PipelineConfig
from src.exceptions import InvalidConfigError
from src.resamplers.adasyn_resampler import ADASYNResampler
from src.resamplers.base import BaseResampler
from src.resamplers.smote_resampler import SMOTEResampler
from src.resamplers.smote_tomek_resampler import SMOTETomekResampler


class ResamplerFactory:
    """Instantiates concrete resamplers using Strategy and Factory design patterns."""

    @staticmethod
    def create(config: PipelineConfig) -> BaseResampler:
        """Builds and returns the configured BaseResampler strategy.

        Args:
            config: Pipeline configuration specifying method and hyperparameters.

        Returns:
            BaseResampler: Initialized resampler strategy.

        Raises:
            InvalidConfigError: If method is unrecognized.
        """
        method = config.method.upper().replace("_", "-")
        strategy = config.resolved_sampling_strategy()

        if method == "SMOTE":
            return SMOTEResampler(
                k_neighbors=config.k_neighbors,
                sampling_strategy=strategy,
                random_state=config.random_seed,
            )

        if method == "ADASYN":
            return ADASYNResampler(
                n_neighbors=config.k_neighbors,
                sampling_strategy=strategy,
                random_state=config.random_seed,
            )

        if method in ["SMOTE-TOMEK", "SMOTETOMEK"]:
            return SMOTETomekResampler(
                k_neighbors=config.k_neighbors,
                sampling_strategy=strategy,
                random_state=config.random_seed,
                tomek_sampling_strategy=config.tomek_sampling_strategy,
            )

        raise InvalidConfigError(
            f"Unknown resampling method '{config.method}'. "
            f"Supported methods: ['SMOTE', 'ADASYN', 'SMOTE-TOMEK']."
        )
