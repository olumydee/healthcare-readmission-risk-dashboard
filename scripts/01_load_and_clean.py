import pandas as pd
from pathlib import Path

# 1) Build file paths relative to this project folder (so it works anywhere)
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "diabetic_data.csv"
OUT_FILE = PROJECT_DIR / "data" / "processed" / "diabetes_clean.csv"

# 2) Read the CSV into a DataFrame (a table in memory)
df = pd.read_csv(RAW_FILE)

# 3) Replace the dataset's missing-value marker "?" with real missing values
df = df.replace("?", pd.NA)

# 4) Create a clean target column:
# readmitted values are: "<30", ">30", "NO"
# We set flag = 1 only for "<30" (readmitted within 30 days)
df["readmitted_30d_flag"] = (df["readmitted"] == "<30").astype(int)

# 5) Save the cleaned dataset to data/processed
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

# 6) Print quick confirmation so you know it worked
print("Saved:", OUT_FILE)
print("Rows:", len(df))
print("Columns:", df.shape[1])
print("30-day readmission rate:", round(df["readmitted_30d_flag"].mean(), 4))
