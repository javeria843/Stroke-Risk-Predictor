"""
Run this ONCE to generate the correct model files.
It will create: model_calibrated.pkl, encoder.pkl, scaler.pkl, feature_names.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# ── Load dataset ─────────────────────────────────────────────────────────────
import os
if os.path.exists("stroke.csv"):
    df = pd.read_csv("stroke.csv")
elif os.path.exists("stroke.xlsx"):
    df = pd.read_excel("stroke.xlsx", engine="openpyxl")
elif os.path.exists("stroke.xls"):
    # Try CSV first (file might be CSV with .xls extension)
    try:
        df = pd.read_csv("stroke.xls")
        print("Loaded stroke.xls as CSV format")
    except Exception:
        try:
            df = pd.read_excel("stroke.xls", engine="xlrd")
        except Exception:
            df = pd.read_excel("stroke.xls", engine="openpyxl")
else:
    raise FileNotFoundError("Dataset not found! Put stroke.csv or stroke.xls in this folder.")

print(f"Dataset loaded: {df.shape}")

# ── Preprocessing (exact same as notebook) ──────────────────────────────────
df = df.drop("id", axis=1)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

X = df.drop("stroke", axis=1)
y = df["stroke"]

cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
num_cols = ['age', 'avg_glucose_level', 'bmi']

# Also keep binary cols (hypertension, heart_disease) as-is
binary_cols = ['hypertension', 'heart_disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Encoder ─────────────────────────────────────────────────────────────────
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[cat_cols])

X_train_cat = pd.DataFrame(
    ohe.transform(X_train[cat_cols]),
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_train.index
)
X_test_cat = pd.DataFrame(
    ohe.transform(X_test[cat_cols]),
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_test.index
)

# ── Scaler ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train_num = pd.DataFrame(
    scaler.transform(X_train[num_cols]),
    columns=num_cols, index=X_train.index
)
X_test_num = pd.DataFrame(
    scaler.transform(X_test[num_cols]),
    columns=num_cols, index=X_test.index
)

# ── Binary cols ──────────────────────────────────────────────────────────────
X_train_bin = X_train[binary_cols].reset_index(drop=False).set_index(X_train.index)
X_test_bin  = X_test[binary_cols].reset_index(drop=False).set_index(X_test.index)

# ── Final feature matrix: [num | binary | cat] ───────────────────────────────
X_train_final = pd.concat([X_train_num, X_train[binary_cols], X_train_cat], axis=1)
X_test_final  = pd.concat([X_test_num,  X_test[binary_cols],  X_test_cat],  axis=1)

feature_names = X_train_final.columns.tolist()
print(f"Features ({len(feature_names)}): {feature_names}")

# ── SMOTE ────────────────────────────────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_final, y_train)

# ── Random Forest ─────────────────────────────────────────────────────────────
best_model = RandomForestClassifier(n_estimators=200, random_state=42)
best_model.fit(X_train_res, y_train_res)
print(f"RF trained. Test accuracy: {best_model.score(X_test_final, y_test):.4f}")

# ── Calibrated model ──────────────────────────────────────────────────────────
calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=5)
calibrated.fit(X_train_res, y_train_res)
print("Calibration done.")

# ── Save all artifacts ────────────────────────────────────────────────────────
with open("model_calibrated.pkl", "wb") as f:
    pickle.dump(calibrated, f)

with open("model_rf.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(ohe, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n✅ All files saved:")
print("  model_calibrated.pkl")
print("  model_rf.pkl")
print("  encoder.pkl")
print("  scaler.pkl")
print("  feature_names.pkl")
