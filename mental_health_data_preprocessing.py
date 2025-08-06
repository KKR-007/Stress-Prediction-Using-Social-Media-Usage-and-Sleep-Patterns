# Imports
import pandas as pd
from pathlib import Path

# 1. File Paths
first_data_path = Path("mental_health_data_final_data.csv")
second_data_path   = Path("mental_health_and_technology_usage_data.csv")

# 2. Load CSVs
first_dataset = pd.read_csv(first_data_path)
second_dataset = pd.read_csv(second_data_path)

# 3. Rename Shared Columns
rename_map = {
    "Social_Media_Usage":        "Social_Media_Usage",
    "Social_Media_Usage_Hours":  "Social_Media_Usage",
    "Sleep_Hours":               "Sleep_Hours",
    "Stress_Level":              "Stress_Level"
}
first_dataset = first_dataset.rename(columns=rename_map)
second_dataset = second_dataset.rename(columns=rename_map)

# 4. Merge Columns
def collapse_mental_health(row):
    status = str(row.get("Mental_Health_Status", "")).strip().lower()
    cond   = str(row.get("Mental_Health_Condition", "")).strip().lower()

    if status in {"good", "fair", "excellent"}:
        return "No"
    if status == "poor":
        return "Yes"
    if cond.startswith("yes"):
        return "Yes"
    if cond.startswith("no"):
        return "No"
    return "Unknown"

for df in (first_dataset, second_dataset):
    df["Mental_Health_Condition"] = df.apply(collapse_mental_health, axis=1)
    df.drop(columns=[col for col in ["Mental_Health_Status", "Mental_Health_Condition"] if col in df.columns and col != "Mental_Health_Condition"],
            inplace=True)

# 5. Align Schemas by Union of Columns
all_columns = sorted(set(first_dataset.columns) | set(second_dataset.columns))
first_dataset = first_dataset.reindex(columns=all_columns)
second_dataset = second_dataset.reindex(columns=all_columns)

# 6. Concatenate Datasets
combined = pd.concat([first_dataset, second_dataset], axis=0, ignore_index=True, sort=False)

# 7. Remove Duplicate Variants of Core Columns
def collapse_duplicates(base_col):
    matches = [col for col in combined.columns if col.lower().startswith(base_col.lower())]
    if len(matches) > 1:
        combined[base_col] = combined[matches].bfill(axis=1).iloc[:, 0]
        combined.drop(columns=[col for col in matches if col != base_col], inplace=True)
    elif len(matches) == 1 and matches[0] != base_col:
        combined.rename(columns={matches[0]: base_col}, inplace=True)

collapse_duplicates("Sleep_Hours")
collapse_duplicates("Stress_Level")
collapse_duplicates("Social_Media_Usage")

# 8. Drop Irrelevant Columns
columns_to_drop = [
    "Alcohol_Consumption", "Consultation_History", "Country", "Diet_Quality", "Gaming_Hours",
    "Medication_Usage", "Occupation", "Online_Support_Usage", "Screen_Time_Hours", "Severity",
    "Smoking_Habit", "Support_Systems_Access", "Technology_Usage_Hours", "Work_Environment_Impact", "Work_Hours"
]
combined.drop(columns=[col for col in columns_to_drop if col in combined.columns], inplace=True)

# 9. Drop Rows with Missing Values
essential_columns = ["Sleep_Hours", "Stress_Level", "Social_Media_Usage",
                     "Mental_Health_Condition", "Age", "Gender", "User_ID","Physical_Activity_Hours"]
combined.dropna(subset=essential_columns, inplace=True)

# 10. Remove Invalid Values
combined = combined[
    (combined["Sleep_Hours"] <= 24) &
    (combined["Social_Media_Usage"] <= 24) &
    (combined["Age"] != 0)
]

# 11. Save Final Cleaned Dataset
final_path = Path("stress_prediction_data_merged.csv")
combined.to_csv(final_path, index=False)
print(f"Cleaned combined dataset saved as: {final_path}")