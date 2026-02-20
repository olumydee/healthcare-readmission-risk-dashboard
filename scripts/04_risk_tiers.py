import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1) Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT_DIR / "data" / "processed" / "features.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "scored_data.csv"

# 2) Load features
df = pd.read_csv(FEATURES_FILE)

target = "readmitted_30d_flag"
X = df.drop(columns=[target, "encounter_id"])
y = df[target]

categorical_cols = ["race", "gender", "age", "insulin", "diabetesMed"]
numeric_cols = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

model = LogisticRegression(max_iter=1000)

clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

# 3) Train on full dataset (since now we are creating scoring output)
clf.fit(X, y)

# 4) Predict probabilities
df["predicted_probability"] = clf.predict_proba(X)[:, 1]

# 5) Create risk tiers based on probability percentiles
df["risk_tier"] = pd.qcut(
    df["predicted_probability"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# 6) Save scored dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Saved scored dataset:", OUTPUT_FILE)
print("\nRisk Tier Distribution:")
print(df["risk_tier"].value_counts())
print("\nReadmission Rate by Risk Tier:")
print(df.groupby("risk_tier")[target].mean())
