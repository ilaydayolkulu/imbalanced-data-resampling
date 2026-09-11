"""Configuration data models and command-line argument parser for the tabular resampling pipeline."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.exceptions import InvalidConfigError


@dataclass
class PipelineConfig:
    """Strongly-typed pipeline configuration parameters."""
    input_path: Path
    output_path: Path
    target_col: str
    method: str
    k_neighbors: int = 5
    sampling_strategy: Union[str, float] = "auto"
    sampling_ratio: Optional[float] = None
    random_seed: int = 42
    tomek_sampling_strategy: str = "auto"

    def resolved_sampling_strategy(self) -> Union[str, float]:
        """Resolves whether to use categorical strategy ('auto', 'minority') or float ratio.

        Returns:
            Union[str, float]: The exact sampling strategy argument accepted by imblearn.
        """
        # If user explicitly passed a numeric sampling_ratio (e.g. 0.8, 1.0) and strategy is 'auto'
        if self.sampling_ratio is not None and self.sampling_strategy == "auto":
            return float(self.sampling_ratio)

        # If sampling_strategy itself is a numeric string (e.g. '0.75')
        if isinstance(self.sampling_strategy, str):
            try:
                val = float(self.sampling_strategy)
                return val
            except ValueError:
                return self.sampling_strategy.strip().lower()

        return self.sampling_strategy


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with Qt/C++ compatible parameter contracts.

    Returns:
        argparse.ArgumentParser: Ready-to-parse argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generic Tabular Imbalanced Data Resampling Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Mandatory path to output CSV file or directory where balanced dataset will be saved with 'resampled_' prefix.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        required=True,
        help="Column name representing target class / label (e.g., 'target' or 'label').",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="SMOTE",
        choices=["SMOTE", "ADASYN", "SMOTE-TOMEK"],
        help="Resampling method to apply: SMOTE, ADASYN, or SMOTE-TOMEK.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors for synthetic sample generation (k-NN).",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="auto",
        help="imblearn sampling strategy: 'auto', 'minority', 'not majority', 'all', or a float ratio string.",
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=None,
        help="Target ratio of minority to majority class (0.0 < ratio <= 1.0). Overrides 'auto' strategy.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Seed for pseudo-random number generator for reproducible sampling.",
    )
    parser.add_argument(
        "--tomek_sampling_strategy",
        type=str,
        default="auto",
        choices=["auto", "all", "not minority", "not_minority"],
        help="Undersampling strategy for Tomek Links cleaning (only applicable to SMOTE-TOMEK).",
    )

    return parser


def parse_and_validate_args(args_list: Optional[list] = None) -> PipelineConfig:
    """Parses command-line arguments and validates structural integrity.

    Args:
        args_list: Optional list of argument strings (useful for testing). If None, sys.argv is used.

    Returns:
        PipelineConfig: Validated pipeline configuration.

    Raises:
        InvalidConfigError: If argument values are logically invalid.
    """
    parser = build_arg_parser()
    parsed = parser.parse_args(args_list)

    # Normalize method name
    norm_method = parsed.method.upper().replace("_", "-")
    if norm_method == "SMOTETOMEK":
        norm_method = "SMOTE-TOMEK"

    if norm_method not in ["SMOTE", "ADASYN", "SMOTE-TOMEK"]:
        raise InvalidConfigError(f"Unsupported method '{parsed.method}'. Choose from SMOTE, ADASYN, SMOTE-TOMEK.")

    # Validate k_neighbors
    if parsed.k_neighbors <= 0:
        raise InvalidConfigError(f"k_neighbors must be a positive integer, got {parsed.k_neighbors}.")

    # Validate sampling_ratio if provided
    if parsed.sampling_ratio is not None:
        if parsed.sampling_ratio <= 0.0 or parsed.sampling_ratio > 1.0:
            raise InvalidConfigError(
                f"sampling_ratio must be in range (0.0, 1.0], got {parsed.sampling_ratio}."
            )

    # Normalize input path
    input_path = Path(parsed.input_path).resolve()

    # Validate mandatory output_path
    if not parsed.output_path or not parsed.output_path.strip():
        raise InvalidConfigError("output_path is required.")

    raw_output_str = parsed.output_path.strip()
    raw_output = Path(raw_output_str)

    # Check if raw_output specifies a directory
    is_directory = (
        raw_output.is_dir()
        or raw_output_str.endswith(("/", "\\"))
        or not raw_output.suffix
    )

    method_tag = norm_method.replace("-", "_").lower()

    if is_directory:
        resolved_dir = raw_output.resolve()
        final_filename = f"resampled_{method_tag}_{input_path.name}"
        output_path = resolved_dir / final_filename
    else:
        parent_dir = raw_output.parent.resolve()
        filename = raw_output.name
        if not filename.startswith("resampled_"):
            filename = f"resampled_{filename}"
        output_path = parent_dir / filename

    # Normalize tomek_sampling_strategy
    tomek_strat = parsed.tomek_sampling_strategy.replace("_", " ")

    return PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        target_col=parsed.target_col.strip(),
        method=norm_method,
        k_neighbors=parsed.k_neighbors,
        sampling_strategy=parsed.sampling_strategy,
        sampling_ratio=parsed.sampling_ratio,
        random_seed=parsed.random_seed,
        tomek_sampling_strategy=tomek_strat,
    )
