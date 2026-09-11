"""Custom domain exceptions and process exit codes for the tabular resampling pipeline."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Standard exit codes for C++ / Qt QProcess communication."""
    SUCCESS = 0
    FILE_IO_ERROR = 1
    INVALID_ARGUMENTS = 2
    DATA_VALIDATION_ERROR = 3
    RESAMPLING_FAILED = 4
    UNEXPECTED_ERROR = 5


class ResamplingError(Exception):
    """Base exception for all tabular resampling pipeline errors."""

    def __init__(self, message: str, exit_code: ExitCode = ExitCode.UNEXPECTED_ERROR):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class FileIOError(ResamplingError):
    """Raised when input file is missing or output path is unwriteable."""

    def __init__(self, message: str):
        super().__init__(message, exit_code=ExitCode.FILE_IO_ERROR)


class InvalidConfigError(ResamplingError):
    """Raised when command-line arguments or configurations are invalid."""

    def __init__(self, message: str):
        super().__init__(message, exit_code=ExitCode.INVALID_ARGUMENTS)


class DataValidationError(ResamplingError):
    """Raised when data structure, target column, or sample sizes fail validation."""

    def __init__(self, message: str):
        super().__init__(message, exit_code=ExitCode.DATA_VALIDATION_ERROR)


class ResamplingExecutionError(ResamplingError):
    """Raised when imblearn algorithms encounter algorithmic or runtime errors."""

    def __init__(self, message: str):
        super().__init__(message, exit_code=ExitCode.RESAMPLING_FAILED)
