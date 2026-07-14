# run_pipeline.py
# Runs the full RiskAtlas production refresh from raw market data
# through Random Forest V3 predictions.

import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PYTHON_STEPS = [
    (
        PROJECT_ROOT / "src/data/stock_load.py",
        "Refreshing market price data",
    ),
    (
        PROJECT_ROOT / "src/data/context_load.py",
        "Refreshing market and company context",
    ),
]

SQL_FILES = [
    PROJECT_ROOT / "sql/staging/stg_market_prices.sql",
    PROJECT_ROOT / "sql/features/price_features.sql",
    PROJECT_ROOT / "sql/features/inference_engineering_v3.sql",
]

PREDICTION_SCRIPT = (
    PROJECT_ROOT / "src/models/prediction_v3.py"
)


def run_command(
    command: list[str],
    step_name: str,
) -> None:
    """Run a shell command and stop the pipeline if it fails."""
    print(f"\n{step_name}")
    print("=" * 60)

    start_time = time.time()

    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )

    except FileNotFoundError as error:
        print(
            f"\nCommand not found while running: "
            f"{step_name}"
        )
        raise SystemExit(1) from error

    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline failed during: "
            f"{step_name}"
        )
        raise SystemExit(
            error.returncode
        ) from error

    elapsed_seconds = time.time() - start_time

    print(
        f"\nCompleted in "
        f"{elapsed_seconds:.1f} seconds"
    )


def run_python_script(
    script_path: Path,
    step_name: str,
) -> None:
    """Run a Python script with the active virtual environment."""
    if not script_path.exists():
        raise FileNotFoundError(
            f"Python script not found: {script_path}"
        )

    run_command(
        [
            sys.executable,
            str(script_path),
        ],
        step_name,
    )


def build_psql_command(
    sql_path: Path,
) -> list[str]:
    """Build the correct psql command for local or cloud PostgreSQL."""
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return [
            "psql",
            database_url,
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            str(sql_path),
        ]

    return [
        "psql",
        "-d",
        "risk_atlas",
        "-U",
        "nathanho",
        "-h",
        "localhost",
        "-p",
        "5432",
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(sql_path),
    ]


def run_sql_file(
    sql_path: Path,
) -> None:
    """Execute one SQL file through psql."""
    if not sql_path.exists():
        raise FileNotFoundError(
            f"SQL file not found: {sql_path}"
        )

    run_command(
        build_psql_command(sql_path),
        f"Running {sql_path.name}",
    )


def main() -> None:
    """Run the complete RiskAtlas production pipeline."""
    print("\nStarting RiskAtlas Production Pipeline")
    print("=" * 60)

    pipeline_start = time.time()

    try:
        for script_path, step_name in PYTHON_STEPS:
            run_python_script(
                script_path,
                step_name,
            )

        for sql_file in SQL_FILES:
            run_sql_file(sql_file)

        run_python_script(
            PREDICTION_SCRIPT,
            "Generating Random Forest V3 predictions",
        )

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        raise SystemExit(130)

    total_seconds = time.time() - pipeline_start

    print("\nRiskAtlas Pipeline Completed Successfully")
    print("=" * 60)
    print(
        f"Total runtime: "
        f"{total_seconds / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()