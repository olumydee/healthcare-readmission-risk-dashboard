import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1) Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT_DIR / "data" / "processed" / "features.csv"

# 2) Load data
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

# 3) Preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# 4) Train logistic regression on full dataset
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

clf.fit(X, y)

# 5) Get feature names after one-hot encoding
ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
encoded_cat = ohe.get_feature_names_out(categorical_cols)
feature_names = list(encoded_cat) + numeric_cols

# 6) Extract coefficients and convert to odds ratios
coefs = clf.named_steps["model"].coef_[0]
odds_ratios = np.exp(coefs)

effects = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "odds_ratio": odds_ratios
})

# 7) Sort: odds_ratio high means increases odds; low means decreases odds
effects_sorted = effects.sort_values("odds_ratio", ascending=False)

print("\nTop 15 features increasing readmission odds:")
print(effects_sorted.head(15).to_string(index=False))

print("\nTop 15 features decreasing readmission odds:")
print(effects_sorted.tail(15).to_string(index=False))
