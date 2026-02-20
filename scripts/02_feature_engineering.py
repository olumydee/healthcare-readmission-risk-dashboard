import pandas as pd
from pathlib import Path

# 1) Define project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "diabetes_clean.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "features.csv"

# 2) Load cleaned dataset
df = pd.read_csv(INPUT_FILE, low_memory=False)

# 3) Select relevant columns for modeling & dashboard
selected_columns = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "insulin",
    "diabetesMed",
    "readmitted_30d_flag"
]

df = df[selected_columns]

# 4) Convert numeric columns explicitly (good practice)
numeric_cols = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5) Handle missing values simply for now
df = df.dropna()

# 6) Save feature dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Saved features file:", OUTPUT_FILE)
print("Rows after cleaning:", len(df))
print("Columns:", df.shape[1])
