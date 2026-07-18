import logging
import sys
from pathlib import Path

import great_expectations as gx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data_raw")
DATA_FILE = DATA_DIR / "bank-additional.csv"
VALIDATION_FLAG = Path("validation_passed.txt")


def run_validation() -> bool:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    context = gx.get_context(mode="ephemeral")

    ds = context.data_sources.add_pandas_filesystem(
        name="bank_raw_data", base_directory=str(DATA_DIR)
    )
    asset = ds.add_csv_asset(name="bank_additional_asset")
    batch_def = asset.add_batch_definition_path(name="bank_batch", path=DATA_FILE.name)

    suite = context.suites.add(gx.ExpectationSuite(name="bank_expectations"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age",
            min_value=18,
            max_value=120,
        )
    )
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="y"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="y", value_set=["yes", "no"])
    )

    val_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="bank_checks", data=batch_def, suite=suite)
    )
    result = val_def.run()
    logger.info("Validation passed: %s", result.success)
    return bool(result.success)


if __name__ == "__main__":
    try:
        passed = run_validation()
        if passed:
            VALIDATION_FLAG.write_text("PASSED", encoding="utf-8")
            sys.exit(0)

        logger.error("Data quality checks failed. Stopping training pipeline.")
        sys.exit(1)
    except Exception:
        logger.exception("Validation stage failed unexpectedly")
        sys.exit(1)
