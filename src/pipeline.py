"""Pipeline orchestrator coordinating dataset ingestion, resampling, validation, and export."""

from datetime import datetime
from pathlib import Path
import time
import traceback
from typing import Any, Dict

from src.config import PipelineConfig
from src.io_handler import IOHandler
from src.logger import setup_logger
from src.report_generator import ExecutionReporter
from src.resamplers.factory import ResamplerFactory


class ResamplingPipeline:
    """Executes end-to-end tabular imbalanced data resampling pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initializes pipeline with validated configuration.

        Args:
            config: Pipeline configuration instance.
        """
        self.config = config
        log_file = self.config.output_dir / "resampling_execution.log"
        self.logger = setup_logger("ResamplingPipeline", log_file_path=log_file)

    def run(self) -> Dict[str, Any]:
        """Executes the complete ingestion -> validation -> resampling -> export cycle.

        Returns:
            Dict[str, Any]: Summary execution metrics and dataset statistics.
        """
        start_datetime = datetime.now()
        start_time = time.perf_counter()
        failing_phase = "Initialization"

        self.logger.info("=" * 70)
        self.logger.info("Generic Tabular Imbalanced Data Resampling Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"Input file : {self.config.input_path}")
        self.logger.info(f"Output file: {self.config.output_path}")
        self.logger.info(f"Output dir : {self.config.output_dir}")
        self.logger.info(f"Target col : {self.config.target_col}")
        self.logger.info(f"Method     : {self.config.method}")
        self.logger.info(f"k_neighbors: {self.config.k_neighbors}")
        self.logger.info(f"Random seed: {self.config.random_seed}")

        try:
            # 1. Load dataset
            failing_phase = "Ingestion (CSV Read)"
            self.logger.info("Loading input dataset...")
            t_load_start = time.perf_counter()
            df = IOHandler.load_dataset(self.config.input_path)
            load_duration = time.perf_counter() - t_load_start
            self.logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns.")

            # 2. Validate and split features & target
            failing_phase = "Validation & Feasibility"
            self.logger.info("Validating schema and class distribution...")
            t_val_start = time.perf_counter()
            X, y, initial_counts = IOHandler.validate_and_split(
                df=df,
                target_col=self.config.target_col,
                k_neighbors=self.config.k_neighbors,
            )
            val_duration = time.perf_counter() - t_val_start

            initial_imbalance_ratio = max(initial_counts.values()) / max(1, min(initial_counts.values()))
            self.logger.info(f"Initial class distribution: {initial_counts}")
            self.logger.info(f"Initial imbalance ratio   : {initial_imbalance_ratio:.2f}:1")

            # 3. Create resampler
            failing_phase = "Algorithm Instantiation"
            resampler = ResamplerFactory.create(self.config)
            self.logger.info(f"Applying algorithm: {resampler.name}...")

            # 4. Execute resampling
            failing_phase = "Algorithmic Resampling"
            resample_start = time.perf_counter()
            X_resampled, y_resampled = resampler.resample(X, y)
            resample_duration = time.perf_counter() - resample_start

            # 5. Compute resampled metrics
            final_counts = y_resampled.value_counts().to_dict()
            final_imbalance_ratio = max(final_counts.values()) / max(1, min(final_counts.values()))

            self.logger.info(f"Resampling completed in {resample_duration:.3f} seconds.")
            self.logger.info(f"Final class distribution  : {final_counts}")
            self.logger.info(f"Final imbalance ratio     : {final_imbalance_ratio:.2f}:1")

            # 6. Save dataset
            failing_phase = "Dataset Serialization (CSV Write)"
            self.logger.info(f"Writing balanced dataset to '{self.config.output_path}'...")
            t_save_start = time.perf_counter()
            IOHandler.save_dataset(
                X=X_resampled,
                y=y_resampled,
                target_col=self.config.target_col,
                output_path=self.config.output_path,
            )
            save_duration = time.perf_counter() - t_save_start

            total_duration = time.perf_counter() - start_time
            end_datetime = datetime.now()

            self.logger.info(f"File successfully written ({len(X_resampled):,} rows).")
            self.logger.info(f"Total pipeline execution time: {total_duration:.3f} seconds.")

            # 7. Generate execution report
            failing_phase = "Report Generation"
            initial_stats = {
                "total_rows": len(df),
                "total_cols": len(df.columns),
                "class_counts": initial_counts,
                "imbalance_ratio": initial_imbalance_ratio,
            }
            final_stats = {
                "total_rows": len(X_resampled),
                "total_cols": len(X_resampled.columns) + 1,
                "class_counts": final_counts,
                "imbalance_ratio": final_imbalance_ratio,
            }
            timings = {
                "total_duration": total_duration,
                "load_duration": load_duration,
                "validation_duration": val_duration,
                "resampling_duration": resample_duration,
                "save_duration": save_duration,
            }
            method_stats = getattr(resampler, "last_execution_stats", {})

            report_path = ExecutionReporter.generate_success_report(
                config=self.config,
                timings=timings,
                initial_stats=initial_stats,
                final_stats=final_stats,
                method_stats=method_stats,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
            self.logger.info(f"Detailed execution report saved to: '{report_path}'")
            self.logger.info("=" * 70)

            return {
                "status": "SUCCESS",
                "method": self.config.method,
                "input_file": str(self.config.input_path),
                "output_file": str(self.config.output_path),
                "output_dir": str(self.config.output_dir),
                "report_file": str(report_path),
                "log_file": str(self.config.output_dir / "resampling_execution.log"),
                "initial_samples": len(df),
                "final_samples": len(X_resampled),
                "initial_counts": initial_counts,
                "final_counts": final_counts,
                "total_duration_sec": total_duration,
            }

        except Exception as exc:
            end_datetime = datetime.now()
            self.logger.error(f"Pipeline crashed during phase '{failing_phase}': {str(exc)}")
            tb_str = traceback.format_exc()
            try:
                fail_report = ExecutionReporter.generate_failure_report(
                    config=self.config,
                    error=exc,
                    traceback_str=tb_str,
                    failing_phase=failing_phase,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                )
                self.logger.info(f"Failure audit report saved to: '{fail_report}'")
            except Exception as rep_exc:
                self.logger.warning(f"Could not generate failure report: {rep_exc}")
            raise
