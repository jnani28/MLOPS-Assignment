"""Generate a simple Evidently data drift report for training vs current data."""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = Path("data_raw") / "bank-additional.csv"
CURRENT_PATH = Path("data_raw") / "bank-additional.csv"
OUTPUT_DIR = Path("reports")
OUTPUT_FILE = OUTPUT_DIR / "data_drift_report.html"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_PATH)

    data_definition = DataDefinition(classification=["y"])
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference_df, data_definition=data_definition),
        current_data=Dataset.from_pandas(current_df, data_definition=data_definition),
    )
    snapshot.save_html(str(OUTPUT_FILE))
    print(f"Evidently report generated at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
