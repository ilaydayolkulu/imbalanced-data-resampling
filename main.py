"""Main CLI entry point for the Generic Tabular Imbalanced Data Resampling Pipeline.

This script can be executed standalone or spawned by C++ / Qt QProcess instances.
Process exit codes indicate success (0) or failure (>0).
"""

from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Optional

from src.config import PipelineConfig, parse_and_validate_args
from src.exceptions import ExitCode, ResamplingError
from src.logger import attach_file_handler, setup_logger
from src.pipeline import ResamplingPipeline
from src.report_generator import ExecutionReporter


def _resolve_fallback_output_dir() -> Path:
    """Attempts to extract the destination output directory from CLI arguments upon failure."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--output_path" and idx + 1 < len(sys.argv):
            candidate_str = sys.argv[idx + 1].strip()
            candidate = Path(candidate_str)
            if candidate.is_dir() or candidate_str.endswith(("/", "\\")) or not candidate.suffix:
                return candidate.resolve()
            return candidate.parent.resolve()
    return Path(".").resolve()


def main() -> int:
    """Executes the resampling CLI application.

    Returns:
        int: Process exit code matching ExitCode enumeration.
    """
    start_datetime = datetime.now()
    logger = setup_logger("ResamplingEngine")
    config: Optional[PipelineConfig] = None
    pipeline_started = False

    try:
        config = parse_and_validate_args()
        attach_file_handler(logger, config.output_dir / "resampling_execution.log")

        pipeline = ResamplingPipeline(config)
        pipeline_started = True
        pipeline.run()
        return int(ExitCode.SUCCESS)

    except ResamplingError as r_err:
        end_datetime = datetime.now()
        logger.error(f"Execution failed [{r_err.__class__.__name__}]: {r_err.message}")

        if not pipeline_started:
            target_dir = config.output_dir if config else _resolve_fallback_output_dir()
            tb_str = traceback.format_exc()
            try:
                fail_report = ExecutionReporter.generate_failure_report(
                    config=config,
                    error=r_err,
                    traceback_str=tb_str,
                    failing_phase="CLI Configuration & Validation",
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    output_dir=target_dir,
                )
                logger.info(f"Failure audit report saved to: '{fail_report}'")
            except Exception as rep_err:
                logger.warning(f"Could not generate failure report: {rep_err}")

        return int(r_err.exit_code)

    except SystemExit as s_err:
        # Raised by argparse --help or invalid CLI args
        code = s_err.code if isinstance(s_err.code, int) else int(ExitCode.INVALID_ARGUMENTS)
        return code

    except Exception as u_err:
        end_datetime = datetime.now()
        logger.critical(f"Unhandled system error: {str(u_err)}")
        traceback.print_exc(file=sys.stderr)

        if not pipeline_started:
            target_dir = config.output_dir if config else _resolve_fallback_output_dir()
            tb_str = traceback.format_exc()
            try:
                fail_report = ExecutionReporter.generate_failure_report(
                    config=config,
                    error=u_err,
                    traceback_str=tb_str,
                    failing_phase="Unhandled Global Exception",
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    output_dir=target_dir,
                )
                logger.info(f"Failure audit report saved to: '{fail_report}'")
            except Exception as rep_err:
                logger.warning(f"Could not generate failure report: {rep_err}")

        return int(ExitCode.UNEXPECTED_ERROR)


if __name__ == "__main__":
    sys.exit(main())
