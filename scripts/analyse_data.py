
import pandas as pd
from pathlib import Path

# Locate the prepared dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "sales_learning_analysis.csv"
)

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=["month"])

# Create before/after academy periods
df["period"] = df["month"].apply(
    lambda x: "Before" if x < pd.Timestamp("2026-04-01")
    else "After"
)

# Summarise overall performance by period
summary = df.groupby("period").agg(
    total_calls=("calls_taken", "sum"),
    total_sales=("sales_made", "sum"),
    total_revenue=("revenue", "sum")
)

# Calculate weighted conversion rates
summary["conversion_rate"] = (
    summary["total_sales"] / summary["total_calls"] * 100
)

print("\nSales performance before and after academy:")
print(summary.round(2))


# Part 2 - Compare performance by training status

training_comparison = df.groupby(
    "trained_before_month"
).agg(
    total_calls=("calls_taken", "sum"),
    total_sales=("sales_made", "sum"),
    total_revenue=("revenue", "sum")
)

training_comparison["conversion_rate"] = (
    training_comparison["total_sales"]
    / training_comparison["total_calls"]
) * 100

print("\nSales performance by training status:")
print(training_comparison.round(2))


# STEP 3A - Compare sales performance by contact centre

# Load advisor information
advisors = pd.read_csv(
    PROJECT_ROOT / "data" / "advisors.csv"
)

# Connect centre and team details to sales records
df = df.merge(
    advisors[["advisor_id", "centre", "team_id"]],
    on="advisor_id",
    how="left",
    validate="many_to_one"
)

# Calculate conversion rate by centre
centre_summary = df.groupby("centre").agg(
    total_calls=("calls_taken", "sum"),
    total_sales=("sales_made", "sum"),
    total_revenue=("revenue", "sum")
)

centre_summary["conversion_rate"] = (
    centre_summary["total_sales"]
    / centre_summary["total_calls"]
) * 100

print("\nSales performance by contact centre:")
print(centre_summary.round(2))


# STEP 3B - Learning completion by contact centre

learning = pd.read_csv(
    PROJECT_ROOT / "data" / "learning_activity.csv"
)

# Connect learning records to advisor centres
learning = learning.merge(
    advisors[["advisor_id", "centre"]],
    on="advisor_id",
    how="left",
    validate="many_to_one"
)

# Calculate completion rate by centre
learning["completed_flag"] = (
    learning["completed"] == "Yes"
)

centre_learning = (
    learning.groupby("centre")["completed_flag"]
    .mean() * 100
)

print("\nModule completion rate by contact centre:")
print(centre_learning.round(2))
