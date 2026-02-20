import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1) Load features
PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT_DIR / "data" / "processed" / "features.csv"

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
log_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])

# Random Forest
rf_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print("Logistic Regression ROC-AUC:", round(log_auc, 4))
print("Random Forest ROC-AUC:", round(rf_auc, 4))

#----- ADD FEATURE IMPORTANCE 

# Extract feature names after encoding

ohe = rf_model.named_steps["preprocess"].named_transformers_["cat"]
encoded_cat_features = ohe.get_feature_names_out(categorical_cols)

all_features = list(encoded_cat_features) + numeric_cols

importances = rf_model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance_df.head(15))

