import pandas as pd 
from pathlib import Path

#locate project data folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT/"data"

#load datasets
advisors = pd.read_csv(DATA_DIR/"advisors.csv")

learning = pd.read_csv(
    DATA_DIR / "learning_activity.csv",
    parse_dates=["completion_date"]
)

coaching = pd.read_csv(
    DATA_DIR / "coaching_attendance.csv"
)

sales = pd.read_csv(
    DATA_DIR / "sales_performance.csv",
    parse_dates=["month"]
)

print("Northbrook Travel - Data Preparation")

print("\nAdvisors:", len(advisors))
print("Learning records:", len(learning))
print("Coaching records:", len(coaching))
print("Sales records:", len(sales))

####################################
#summarise learning by advisor

advisor_learning = (
    learning.groupby("advisor_id")
    .agg(
        modules_completed = (
            "completed",
            lambda x: (x =="Yes").sum()
        ),
        last_completion_date=(
            "completion_date",
            "max"
        )
    )
    .reset_index()
)

print("\nAdvisor learning summary:")
print(advisor_learning.head(10))

######################################
#connect sales and learning data

sales_with_learning = sales.merge(
    advisor_learning,
    on = "advisor_id",
    how = "left"
)

print("\nSales and learning data connected:")

print(sales_with_learning[[
    "advisor_id",
    "month",
    "conversion_rate",
    "modules_completed",
    "last_completion_date"
]].head(15).to_string(index=False))

print("\nTotal merged records:", len(sales_with_learning))

##################################################
# Calculate training completed by each sales month
# Get the final day of every sales month
sales_with_learning["month_end"] = (
    sales_with_learning["month"]
    + pd.offsets.MonthEnd(0)
)

# Count completed modules up to that month
sales_with_learning["modules_completed_to_date"] = 0

for index, row in sales_with_learning.iterrows():

    completed_modules = learning[
        (learning["advisor_id"] == row["advisor_id"]) &
        (learning["completed"] == "Yes") &
        (learning["completion_date"] <= row["month_end"])
    ]

    sales_with_learning.at[
        index, "modules_completed_to_date"
    ] = len(completed_modules)

# Display example
print("\nTraining progress for Advisor 1:")

print(
    sales_with_learning[
        sales_with_learning["advisor_id"] == 1
    ][[
        "month",
        "modules_completed_to_date"
    ]].to_string(index=False)
)

############################################
# Identify training before each sales month

sales_with_learning["trained_before_month"] = (
    sales_with_learning["modules_completed_to_date"] > 0
)

# Adjust for modules completed during the current month
for index, row in sales_with_learning.iterrows():

    completed_before = learning[
        (learning["advisor_id"] == row["advisor_id"]) &
        (learning["completed"] == "Yes") &
        (learning["completion_date"] < row["month"])
    ]

    sales_with_learning.at[
        index, "trained_before_month"
    ] = len(completed_before) > 0

print("\nTraining status for Advisor 1:")

print(
    sales_with_learning[
        sales_with_learning["advisor_id"] == 1
    ][[
        "month",
        "trained_before_month"
    ]].to_string(index=False)
)

############################
# Simulate a possible training effect

import numpy as np

rng = np.random.default_rng(123)

# Baseline conversion probability: 15%
# Hypothetical training uplift: 1 percentage point
sales_with_learning["simulated_conversion_probability"] = (
    0.15
    + np.where(
        sales_with_learning["trained_before_month"],
        0.01,
        0
    )
)

# Regenerate sales using the simulated probabilities
sales_with_learning["sales_made"] = rng.binomial(
    n=sales_with_learning["calls_taken"].to_numpy(),
    p=sales_with_learning[
        "simulated_conversion_probability"
    ].to_numpy()
)

# Recalculate conversion rate
sales_with_learning["conversion_rate"] = (
    sales_with_learning["sales_made"]
    / sales_with_learning["calls_taken"]
) * 100

print("\nAverage conversion rate by training status:")
print(
    sales_with_learning.groupby(
        "trained_before_month"
    )["conversion_rate"].mean().round(2)
)

###########################
# Recalculate revenue and insurance metrics

# Recalculate revenue using updated sales
sales_with_learning["revenue"] = (
    sales_with_learning["sales_made"]
    * sales_with_learning["average_sale_value"]
)

# Generate insurance sales based on updated sales
sales_with_learning["insurance_sales"] = rng.binomial(
    n=sales_with_learning["sales_made"].to_numpy(),
    p=0.40
)

# Recalculate insurance attach rate
sales_with_learning["insurance_attach_rate"] = (
    sales_with_learning["insurance_sales"]
    / sales_with_learning["sales_made"]
) * 100

# Display updated results
print("\nUpdated sales and revenue:")
print(sales_with_learning[[
    "advisor_id",
    "month",
    "sales_made",
    "revenue",
    "insurance_sales",
    "insurance_attach_rate"
]].head(10).to_string(index=False))


########################################
#Validate prepared sales data

print("\nDATA VALIDATION REPORT")

# 1. Check total records
print("Total records:", len(sales_with_learning))

# 2. Check duplicate advisor-month records
duplicates = sales_with_learning.duplicated(
    subset=["advisor_id", "month"]
).sum()

print("Duplicate advisor-month records:", duplicates)

# 3. Check missing values in important columns
important_columns = [
    "advisor_id",
    "month",
    "calls_taken",
    "sales_made",
    "conversion_rate",
    "revenue",
    "insurance_sales",
    "customer_satisfaction"
]

print("\nMissing values:")
print(sales_with_learning[important_columns].isna().sum())

# 4. Check business rules
print("\nSales greater than calls:",
      (sales_with_learning["sales_made"] >
       sales_with_learning["calls_taken"]).sum())

print("Insurance sales greater than total sales:",
      (sales_with_learning["insurance_sales"] >
       sales_with_learning["sales_made"]).sum())

# 5. Check revenue calculation
revenue_errors = (
    sales_with_learning["revenue"] !=
    sales_with_learning["sales_made"] *
    sales_with_learning["average_sale_value"]
).sum()

print("Revenue calculation errors:", revenue_errors)

##############################################

# Export final prepared dataset

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save merged and validated sales-learning data
sales_with_learning.to_csv(
    OUTPUT_DIR / "sales_learning_analysis.csv",
    index=False
)

print("\nPower BI-ready dataset exported successfully!")

print("File location:",
      OUTPUT_DIR / "sales_learning_analysis.csv")

print("Total exported records:",
      len(sales_with_learning))
