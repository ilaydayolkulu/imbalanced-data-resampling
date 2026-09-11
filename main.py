"""Main CLI entry point for the Generic Tabular Imbalanced Data Resampling Pipeline.

This script can be executed standalone or spawned by C++ / Qt QProcess instances.
Process exit codes indicate success (0) or failure (>0).
"""

import sys
import traceback

from src.config import parse_and_validate_args
from src.exceptions import ExitCode, ResamplingError
from src.logger import setup_logger
from src.pipeline import ResamplingPipeline


def main() -> int:
    """Executes the resampling CLI application.

    Returns:
        int: Process exit code matching ExitCode enumeration.
    """
    logger = setup_logger("ResamplingEngine")

    try:
        config = parse_and_validate_args()
        pipeline = ResamplingPipeline(config)
        pipeline.run()
        return int(ExitCode.SUCCESS)

    except ResamplingError as r_err:
        logger.error(f"Execution failed [{r_err.__class__.__name__}]: {r_err.message}")
        return int(r_err.exit_code)

    except SystemExit as s_err:
        # Raised by argparse --help or invalid CLI args
        code = s_err.code if isinstance(s_err.code, int) else int(ExitCode.INVALID_ARGUMENTS)
        return code

    except Exception as u_err:
        logger.critical(f"Unhandled system error: {str(u_err)}")
        traceback.print_exc(file=sys.stderr)
        return int(ExitCode.UNEXPECTED_ERROR)


if __name__ == "__main__":
    sys.exit(main())
