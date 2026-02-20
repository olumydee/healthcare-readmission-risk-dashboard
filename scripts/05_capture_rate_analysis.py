import pandas as pd
from pathlib import Path

# 1) Load scored dataset (created in script 04)
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "scored_data.csv"

df = pd.read_csv(INPUT_FILE)

# 2) Total readmissions in full dataset
total_readmissions = df["readmitted_30d_flag"].sum()

print("Total patients:", len(df))
print("Total readmissions:", total_readmissions)
print("Base readmission rate:", round(total_readmissions / len(df), 4))

print("\n--- Capture Rate Analysis ---")

# 3) Function to compute capture rate for any top percentage
def compute_capture_rate(top_percent):
    top_n = int(len(df) * top_percent)
    
    top_group = df.sort_values(
        "predicted_probability",
        ascending=False
    ).head(top_n)
    
    readmissions_in_group = top_group["readmitted_30d_flag"].sum()
    
    capture_rate = readmissions_in_group / total_readmissions
    
    print(f"\nTop {int(top_percent*100)}% group size:", top_n)
    print("Readmissions in group:", readmissions_in_group)
    print("Capture Rate:", round(capture_rate, 4))
    print("Readmission rate inside group:",
          round(readmissions_in_group / top_n, 4))


# 4) Test for 10%, 15%, and 20%
compute_capture_rate(0.10)
compute_capture_rate(0.15)
compute_capture_rate(0.20)
