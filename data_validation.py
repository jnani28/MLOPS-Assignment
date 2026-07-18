import sys
import great_expectations as gx

# 1. Get Ephemeral Context
context = gx.get_context(mode="ephemeral")

# 2. Add Pandas datasource pointing to our raw data folder
ds = context.data_sources.add_pandas_filesystem(
    name="bank_raw_data", base_directory="data_raw/")

# 3. Add the CSV file as an asset
asset = ds.add_csv_asset(name="bank_additional_asset")

# 4. Define the batch referencing our dataset
batch_def = asset.add_batch_definition_path(
    name="bank_batch", path="bank-additional.csv")

# 5. Define the expectation suite
suite = context.suites.add(gx.ExpectationSuite(name="bank_expectations"))

# 6. Define Quality Checks (Expectations)
# Check 1: Age must be between 18 and 120
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="age", min_value=18, max_value=120)
)

# Check 2: Target column 'y' should never be null
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="y")
)

# Check 3: Target column 'y' must only contain 'yes' or 'no'
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="y", value_set=["yes", "no"])
)

# 7. Run validation
val_def = context.validation_definitions.add(
    gx.ValidationDefinition(
        name="bank_checks", data=batch_def, suite=suite))
result = val_def.run()

print(f"Validation Passed: {result.success}")

# 8. Handle results for the MLOps pipeline
if result.success:
    # Write a success indicator file for DVC tracking
    with open("validation_passed.txt", "w") as f:
        f.write("PASSED")
    sys.exit(0)
else:
    print("Data quality checks failed! Stopping training pipeline.")
    sys.exit(1)
