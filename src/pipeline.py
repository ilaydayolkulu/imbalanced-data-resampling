"""Pipeline orchestrator coordinating dataset ingestion, resampling, validation, and export."""

import time
from typing import Any, Dict

from src.config import PipelineConfig
from src.io_handler import IOHandler
from src.logger import log_qt_event, setup_logger
from src.resamplers.factory import ResamplerFactory


class ResamplingPipeline:
    """Executes end-to-end tabular imbalanced data resampling pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initializes pipeline with validated configuration.

        Args:
            config: Pipeline configuration instance.
        """
        self.config = config
        self.logger = setup_logger("ResamplingPipeline")

    def run(self) -> Dict[str, Any]:
        """Executes the complete ingestion -> validation -> resampling -> export cycle.

        Returns:
            Dict[str, Any]: Summary execution metrics and dataset statistics.
        """
        start_time = time.perf_counter()

        self.logger.info("=" * 70)
        self.logger.info("Generic Tabular Imbalanced Data Resampling Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"Input file : {self.config.input_path}")
        self.logger.info(f"Output file: {self.config.output_path}")
        self.logger.info(f"Target col : {self.config.target_col}")
        self.logger.info(f"Method     : {self.config.method}")
        self.logger.info(f"k_neighbors: {self.config.k_neighbors}")
        self.logger.info(f"Random seed: {self.config.random_seed}")

        log_qt_event("PIPELINE_STARTED", {
            "input_path": str(self.config.input_path),
            "output_path": str(self.config.output_path),
            "method": self.config.method,
        })

        # 1. Load dataset
        self.logger.info("Loading input dataset...")
        df = IOHandler.load_dataset(self.config.input_path)
        self.logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns.")

        # 2. Validate and split features & target
        self.logger.info("Validating schema and class distribution...")
        X, y, initial_counts = IOHandler.validate_and_split(
            df=df,
            target_col=self.config.target_col,
            k_neighbors=self.config.k_neighbors,
        )

        initial_imbalance_ratio = max(initial_counts.values()) / max(1, min(initial_counts.values()))
        self.logger.info(f"Initial class distribution: {initial_counts}")
        self.logger.info(f"Initial imbalance ratio   : {initial_imbalance_ratio:.2f}:1")

        log_qt_event("DATASET_VALIDATED", {
            "initial_counts": {str(k): int(v) for k, v in initial_counts.items()},
            "imbalance_ratio": round(initial_imbalance_ratio, 2),
            "features_count": X.shape[1],
        })

        # 3. Create resampler
        resampler = ResamplerFactory.create(self.config)
        self.logger.info(f"Applying algorithm: {resampler.name}...")

        # 4. Execute resampling
        resample_start = time.perf_counter()
        X_resampled, y_resampled = resampler.resample(X, y)
        resample_duration = time.perf_counter() - resample_start

        # 5. Compute resampled metrics
        final_counts = y_resampled.value_counts().to_dict()
        final_imbalance_ratio = max(final_counts.values()) / max(1, min(final_counts.values()))

        self.logger.info(f"Resampling completed in {resample_duration:.3f} seconds.")
        self.logger.info(f"Final class distribution  : {final_counts}")
        self.logger.info(f"Final imbalance ratio     : {final_imbalance_ratio:.2f}:1")

        log_qt_event("RESAMPLING_COMPLETED", {
            "final_counts": {str(k): int(v) for k, v in final_counts.items()},
            "final_imbalance_ratio": round(final_imbalance_ratio, 2),
            "duration_seconds": round(resample_duration, 3),
        })

        # 6. Save dataset
        self.logger.info(f"Writing balanced dataset to '{self.config.output_path}'...")
        IOHandler.save_dataset(
            X=X_resampled,
            y=y_resampled,
            target_col=self.config.target_col,
            output_path=self.config.output_path,
        )

        total_duration = time.perf_counter() - start_time
        self.logger.info(f"File successfully written ({len(X_resampled):,} rows).")
        self.logger.info(f"Total pipeline execution time: {total_duration:.3f} seconds.")
        self.logger.info("=" * 70)

        summary = {
            "status": "SUCCESS",
            "method": self.config.method,
            "input_file": str(self.config.input_path),
            "output_file": str(self.config.output_path),
            "initial_samples": len(df),
            "final_samples": len(X_resampled),
            "initial_counts": initial_counts,
            "final_counts": final_counts,
            "total_duration_sec": total_duration,
        }

        log_qt_event("PIPELINE_SUCCESS", summary)
        return summary
