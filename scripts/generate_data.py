import pandas as pd 
import numpy as np
from pathlib import Path 

#random data generation(fictional and synthectic)

SEED = 42
rng = np.random.default_rng(SEED)

N_ADVISORS = 400

CENTRES = ["Leeds", "Glasgow", "Cardiff"]

MODULES = [
    "Needs Discovery",
    "Objection Handling",
    "Upselling Insurance",
    "Compliance Essentials",
    "Customer Empathy"
]

project_root = Path(__file__).resolve().parent.parent #__file__ location of the Python, .resolve() absolute path, 
DATA_DIR = project_root / "data"  #folder named data. Store its location in DATA_DIR.
DATA_DIR.mkdir(parents = True, exist_ok=True)  # if parent folder is missing it will add first them save file

print("Northbrook Travel - Synthetic Data Generator")
print(f"Number of advisors: { N_ADVISORS}")
print(f"Contact centres : {', '.join(CENTRES)}")
print(f"Academy modules: {len(MODULES)}")
print(f"Output directory: {DATA_DIR}")


# 400 fictional advisors
advisors = pd.DataFrame({
    "advisor_id": range(1,401),
    "centre": rng.choice(CENTRES, N_ADVISORS),
    "experience_years" : rng.integers(1,11,N_ADVISORS)
})

#assign advisors to team of 10-15 people
advisors["team_id"] = ""     #creates an empty team column

for centre, group in advisors.groupby("centre"):    #processes Leeds, Glasgow and Cardiff separately

    n_advisors_in_centre = len(group)
    N_TEAMS = round(n_advisors_in_centre/12) #Calculates approximately how many teams are needed
    teams = np.array_split(group.index, N_TEAMS)
#unique team names for each centre
    for team_num, indices in enumerate(teams, start =1):
        advisors.loc[indices, "team_id"] = f"{centre}_Team_{team_num}" #Assigns the team names to advisors


#generate data into csv file
advisors.to_csv(DATA_DIR / "advisors.csv", index = False)

#get 5 records
print(advisors.head())
print("Advisors dataset created successfully.")

print("\nTotal teams: ", advisors["team_id"].nunique())

print("\n Advisors per team:")

# generating learning activity dataset

learning_records = []

for advisor_id in advisors["advisor_id"]:
    for module in MODULES:
        completed = rng.choice(["Yes","No"], p=[0.7, 0.3])

        learning_records.append({
            "advisor_id" : advisor_id,
            "module": module,
            "completed":completed
        })

learning = pd.DataFrame(learning_records)

#generate learning time in minutes
learning["learning_minutes"] = rng.integers(
    15, 91, size=len(learning)
)

#generate dates for completed modules
learning["completion_date"] = pd.NaT 

completed_mask = learning['completed'] == 'Yes'

learning.loc[completed_mask, "completion_date"] = (
    pd.Timestamp("2026-04-01") + pd.to_timedelta(
        rng.integers(0, 183, completed_mask.sum()),
        unit = "D"
    )
)

# get in csv
learning.to_csv(
    DATA_DIR / "learning_activity.csv",
    index=False
)

print("\nLearning Activity dataset:")
print(learning.head(10))

print("\nTotal learning records:", len(learning))

# count completed and incompleted modules
print("\nModule completion summary:")
print(learning['completed'].value_counts())

#module completion rate
completed_count = (learning["completed"]=="Yes").sum()

completion_rate = (
    completed_count /len(learning)
)*100

print(f"\nModule completion rate:{completion_rate:.2f}%")

#calculate completion percentage  for each module
module_completion = (
    learning.groupby("module")['completed']
    .apply(lambda x: (x == 'Yes').mean() * 100)
)

print("\nCompletion rate by module: ")
print(module_completion.round(2))

#generate coaching attendance data

coaching = pd.DataFrame({
    "advisor_id": advisors["advisor_id"],
    "sessions_attended" : rng.integers(
        0,5,size=len(advisors)
    )
})

#Save coaching dataset
coaching.to_csv(
    DATA_DIR / "coaching_attendance.csv",
    index=False
)

print("\n Coaching attendance dataset:")
print(coaching.head(10))

print("\nAverage coaching sessions attended: ")
print(round(coaching["sessions_attended"].mean(), 2))

#generate monthly sales performance records

months = pd.date_range(start = '2025-10-01', periods = 12, freq="MS")

sales_records = []

for advisor_id in advisors["advisor_id"]:
    for month in months:
        sales_records.append({
            'advisor_id' : advisor_id,
            'month': month
        })

sales = pd.DataFrame(sales_records)

#generate monthly calls for each advisor
sales['calls_taken'] = rng.integers(
    150, 301, size=len(sales)   #Randomly generates 150–300 calls per advisor per month
)

#generate sales made from calls
sales['sales_made'] = rng.binomial(  #Simulates how many calls resulted in a sale
    n=sales['calls_taken'], 
    p=0.15
)

#calculate conversion rate
sales["conversion_rate"] = (
    sales["sales_made"]/sales["calls_taken"] #Calculates the proportion of successful calls
)*100

#generate average revenue per sale
sales['average_sale_value'] = rng.integers(
    500, 2001, size=len(sales)
)

#calculate total monthly revenue
sales['revenue'] = (
    sales['sales_made']*sales['average_sale_value']
)

#generate insurance sales
sales['insurance_sales'] = rng.binomial(
    n=sales['sales_made'],
    p=0.40
)

#calculate insurance attach rate
sales['insurance_attach_rate']=(
    sales['insurance_sales']/sales['sales_made']
)*100


# Generate customer satisfaction scores (1 to 5)
sales["customer_satisfaction"] = rng.normal(
    loc=4.2,
    scale=0.4,
    size=len(sales)
)

# Keep scores between 1 and 5
sales["customer_satisfaction"] = (
    sales["customer_satisfaction"]
    .clip(1, 5)
    .round(2)
)

print("\nCustomer satisfaction sample:")
print(sales[[
    "advisor_id",
    "month",
    "customer_satisfaction"
]].head(10))

print("\nSales and revenue sample:")
print(sales.head(5).to_string(index=False))

print("\nSales performance sample:")

print("\nMonthly sales dataset:")
print("\nTotal sales records:", len(sales))

sales.to_csv(
    DATA_DIR / "sales_performance.csv",
    index=False
)

print("Sales performance dataset saved successfully!")