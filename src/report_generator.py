"""Comprehensive execution reporting utility generating structured text and comparative metrics."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import PipelineConfig


class ExecutionReporter:
    """Formats and writes structured execution benchmark and comparison reports."""

    @staticmethod
    def _format_table(headers: list, rows: list) -> str:
        """Formats an aligned ASCII table."""
        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, val in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(str(val)))

        header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
        sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

        table_lines = [sep_line, header_line, sep_line]
        for row in rows:
            table_lines.append("| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)) + " |")
        table_lines.append(sep_line)
        return "\n".join(table_lines)

    @classmethod
    def generate_success_report(
        cls,
        config: PipelineConfig,
        timings: Dict[str, float],
        initial_stats: Dict[str, Any],
        final_stats: Dict[str, Any],
        method_stats: Dict[str, Any],
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> Path:
        """Constructs and writes a detailed comparative execution report.

        Args:
            config: Pipeline configuration instance.
            timings: Phase duration measurements in seconds.
            initial_stats: Initial rows, cols, class distribution, imbalance ratio.
            final_stats: Final rows, cols, class distribution, imbalance ratio.
            method_stats: Resampler-specific internal counters.
            start_datetime: Wall-clock start timestamp.
            end_datetime: Wall-clock completion timestamp.

        Returns:
            Path: Absolute path to the generated report file.
        """
        timestamp_str = end_datetime.strftime("%Y%m%d_%H%M%S")
        method_tag = config.method.lower().replace("-", "_")
        report_filename = f"resampling_report_{method_tag}_{timestamp_str}.txt"
        report_path = config.output_dir / report_filename

        total_duration = timings.get("total_duration", 0.0)
        total_duration_ms = total_duration * 1000.0

        # Phase timing percentages
        load_sec = timings.get("load_duration", 0.0)
        val_sec = timings.get("validation_duration", 0.0)
        resample_sec = timings.get("resampling_duration", 0.0)
        save_sec = timings.get("save_duration", 0.0)

        load_pct = (load_sec / total_duration * 100) if total_duration > 0 else 0.0
        val_pct = (val_sec / total_duration * 100) if total_duration > 0 else 0.0
        resample_pct = (resample_sec / total_duration * 100) if total_duration > 0 else 0.0
        save_pct = (save_sec / total_duration * 100) if total_duration > 0 else 0.0

        # File size formatting
        csv_size_bytes = config.output_path.stat().st_size if config.output_path.exists() else 0
        csv_size_mb = csv_size_bytes / (1024 * 1024)

        # Class distribution table
        init_counts = initial_stats["class_counts"]
        final_counts = final_stats["class_counts"]
        init_total = initial_stats["total_rows"]
        final_total = final_stats["total_rows"]

        all_classes = sorted(list(set(init_counts.keys()).union(set(final_counts.keys()))))
        table_rows = []
        for c in all_classes:
            i_cnt = init_counts.get(c, 0)
            f_cnt = final_counts.get(c, 0)
            i_pct = (i_cnt / init_total * 100) if init_total > 0 else 0.0
            f_pct = (f_cnt / final_total * 100) if final_total > 0 else 0.0
            delta = f_cnt - i_cnt
            delta_str = f"+{delta:,}" if delta > 0 else f"{delta:,}"
            if delta > 0:
                delta_str += " (synth)"
            elif delta < 0:
                delta_str += " (pruned)"
            else:
                delta_str += " (neutral)"

            table_rows.append([
                str(c),
                f"{i_cnt:,} ({i_pct:6.2f}%)",
                f"{f_cnt:,} ({f_pct:6.2f}%)",
                delta_str,
            ])

        net_total_delta = final_total - init_total
        net_total_pct = (net_total_delta / init_total * 100) if init_total > 0 else 0.0
        table_rows.append([
            "Total",
            f"{init_total:,} (100.00%)",
            f"{final_total:,} (100.00%)",
            f"+{net_total_delta:,} (+{net_total_pct:.2f}%)",
        ])

        dist_table = cls._format_table(
            headers=["Class", "Initial (Count, %)", "Final (Count, %)", "Net Delta"],
            rows=table_rows,
        )

        # Method-specific section
        method_upper = config.method.upper()
        if "TOMEK" in method_upper:
            smote_gen = method_stats.get("smote_generated", net_total_delta)
            tomek_pru = method_stats.get("tomek_pruned", 0)
            method_section = (
                f"Stage 1 (SMOTE Generative)  : +{smote_gen:,} synthetic samples synthesized\n"
                f"Stage 2 (Tomek Pruning)     :  {tomek_pru:,} noisy boundary links pruned\n"
                f"Net Resampling Gain         : +{smote_gen - tomek_pru:,} net clean instances added"
            )
        elif method_upper == "ADASYN":
            syn_gen = method_stats.get("synthetic_generated", net_total_delta)
            method_section = (
                f"Adaptive Synthesis Focus    : Borderline density distribution weighting\n"
                f"Total Synthetic Generated   : +{syn_gen:,} adaptive instances synthesized"
            )
        else:
            syn_gen = method_stats.get("synthetic_generated", net_total_delta)
            method_section = (
                f"Uniform k-NN Interpolation  : Standard linear interpolation over feature manifold\n"
                f"Total Synthetic Generated   : +{syn_gen:,} synthetic instances synthesized"
            )

        log_file_path = config.output_dir / "resampling_execution.log"

        lines = [
            "=" * 80,
            "         IMBALANCED TABULAR DATA RESAMPLING - DETAILED EXECUTION REPORT",
            "=" * 80,
            f"Generated Timestamp     : {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            "Pipeline Status         : SUCCESS",
            f"Applied Method          : {config.method}",
            f"Target Column Name      : {config.target_col}",
            "",
            "-" * 80,
            "1. EXECUTION LIFECYCLE & PERFORMANCE PROFILING",
            "-" * 80,
            f"Start Time              : {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Completion Time         : {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Elapsed Duration  : {total_duration:.3f} s ({total_duration_ms:,.2f} ms)",
            "",
            "Granular Phase Timings (time.perf_counter):",
            f"  - 1. Ingestion (CSV Read)       : {load_sec:8.3f} s ({load_sec*1000:9.2f} ms) | {load_pct:5.2f}%",
            f"  - 2. Validation & Feasibility   : {val_sec:8.3f} s ({val_sec*1000:9.2f} ms) | {val_pct:5.2f}%",
            f"  - 3. Algorithmic Resampling     : {resample_sec:8.3f} s ({resample_sec*1000:9.2f} ms) | {resample_pct:5.2f}%",
            f"  - 4. Dataset Serialization (CSV): {save_sec:8.3f} s ({save_sec*1000:9.2f} ms) | {save_pct:5.2f}%",
            "",
            "-" * 80,
            "2. HYPERPARAMETER CONFIGURATION",
            "-" * 80,
            f"Input Dataset Path      : {config.input_path}",
            f"Target Output CSV Path  : {config.output_path}",
            f"Base Output Directory   : {config.output_dir}",
            f"Resampling Algorithm    : {config.method}",
            f"k-Nearest Neighbors (k) : {config.k_neighbors}",
            f"Sampling Strategy       : {config.sampling_strategy}",
            f"Sampling Ratio Override : {config.sampling_ratio if config.sampling_ratio is not None else 'None (auto)'}",
            f"Random Number Seed      : {config.random_seed}",
            f"Tomek Pruning Policy    : {config.tomek_sampling_strategy if 'TOMEK' in method_upper else 'N/A'}",
            "",
            "-" * 80,
            "3. DATASET DISTRIBUTION & CLASS BALANCE METRICS",
            "-" * 80,
            f"Initial Dimensions      : {init_total:,} rows x {initial_stats['total_cols']} columns",
            f"Final Dimensions        : {final_total:,} rows x {final_stats['total_cols']} columns",
            f"Net Rows Change         : +{net_total_delta:,} rows (+{net_total_pct:.2f}%)",
            "",
            "Class Distribution Breakdown:",
            dist_table,
            "",
            f"Imbalance Ratio Delta   : {initial_stats['imbalance_ratio']:.2f}:1  ==>  {final_stats['imbalance_ratio']:.2f}:1",
            "",
            "-" * 80,
            f"4. METHOD-SPECIFIC ALGORITHMIC METRICS ({config.method})",
            "-" * 80,
            method_section,
            "",
            "-" * 80,
            "5. OUTPUT ARTIFACTS MANIFEST",
            "-" * 80,
            f"Balanced Dataset CSV    : {config.output_path}",
            f"CSV File Size           : {csv_size_mb:.2f} MB ({csv_size_bytes:,} bytes)",
            f"Execution Audit Log     : {log_file_path}",
            f"Execution Report File   : {report_path}",
            "=" * 80,
            "",
        ]

        report_content = "\n".join(lines)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_content, encoding="utf-8")
        return report_path

    @classmethod
    def generate_failure_report(
        cls,
        config: Optional[PipelineConfig],
        error: Exception,
        traceback_str: str,
        failing_phase: str,
        start_datetime: datetime,
        end_datetime: datetime,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Constructs and writes an audit failure report when execution crashes.

        Args:
            config: Pipeline configuration if available.
            error: The caught exception instance.
            traceback_str: Full stack trace string.
            failing_phase: Pipeline step where crash occurred.
            start_datetime: Wall-clock start timestamp.
            end_datetime: Crash timestamp.
            output_dir: Target output directory fallback.

        Returns:
            Path: Absolute path to the generated failure report.
        """
        target_dir = (config.output_dir if config else output_dir) or Path(".").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = end_datetime.strftime("%Y%m%d_%H%M%S")
        method_tag = config.method.lower().replace("-", "_") if config else "unknown"
        report_filename = f"resampling_report_{method_tag}_{timestamp_str}_FAILED.txt"
        report_path = target_dir / report_filename

        total_sec = (end_datetime - start_datetime).total_seconds()
        exit_code = getattr(error, "exit_code", 1)

        lines = [
            "=" * 80,
            "         IMBALANCED TABULAR DATA RESAMPLING - EXECUTION FAILURE REPORT",
            "=" * 80,
            f"Failure Timestamp       : {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            "Pipeline Status         : FAILED",
            f"Process Exit Code       : {int(exit_code)}",
            f"Failing Pipeline Phase  : {failing_phase}",
            f"Exception Class         : {error.__class__.__name__}",
            f"Error Summary           : {str(error)}",
            "",
            "-" * 80,
            "1. CONTEXT & LIFECYCLE",
            "-" * 80,
            f"Start Time              : {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Failure Time            : {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration Before Failure : {total_sec:.3f} s ({total_sec * 1000.0:,.2f} ms)",
            f"Configured Input Path   : {config.input_path if config else 'N/A'}",
            f"Configured Output Path  : {config.output_path if config else 'N/A'}",
            f"Configured Method       : {config.method if config else 'N/A'}",
            "",
            "-" * 80,
            "2. ERROR DIAGNOSTICS & TRACEBACK",
            "-" * 80,
            traceback_str.strip(),
            "",
            "=" * 80,
        ]

        report_content = "\n".join(lines)
        report_path.write_text(report_content, encoding="utf-8")
        return report_path
