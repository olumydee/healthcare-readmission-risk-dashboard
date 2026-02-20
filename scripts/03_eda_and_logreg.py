import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# 1) Paths: locate features.csv no matter where you run from
PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT_DIR / "data" / "processed" / "features.csv"

# 2) Load the feature dataset we created in script 02
df = pd.read_csv(FEATURES_FILE)

# 3) Separate predictors (X) from target (y)
target = "readmitted_30d_flag"
X = df.drop(columns=[target, "encounter_id"])  # encounter_id is an ID, not a predictor
y = df[target]

# 4) Define which columns are categorical vs numeric
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

# 5) Preprocess:
# - one-hot encode categorical columns so the model can use them
# - keep numeric columns as they are
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# 6) Define a baseline model (logistic regression)
model = LogisticRegression(max_iter=1000)

# 7) Pipeline = preprocessing + model in one object
clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

# 8) Split data into train and test sets
# stratify=y preserves the same class balance in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9) Train
clf.fit(X_train, y_train)

# 10) Predict probabilities and classes
y_proba = clf.predict_proba(X_test)[:, 1]          # probability of readmission within 30 days
y_pred = (y_proba >= 0.5).astype(int)              # default threshold of 0.5

# 11) Evaluate
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", round(auc, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("\nBase readmission rate:", round(y.mean(), 4))
