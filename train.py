# train.py (optimized for speed)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
import joblib

# -----------------------
# Utility: best threshold
# -----------------------
def best_threshold_from_roc(y_true, y_scores):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr
    ix = np.argmax(j)
    th = thr[ix]
    if not np.isfinite(th):
        th = 0.5
    return float(th)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("heart_2022_no_nans.csv")
y = df["HadHeartAttack"].map({"Yes": 1, "No": 0})
X = df.drop("HadHeartAttack", axis=1)

# -----------------------
# Numeric & categorical columns
# -----------------------
num_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "HeightInMeters", "WeightInKilograms", "BMI"]
cat_cols = [c for c in X.columns if c not in num_cols]

# -----------------------
# Preprocessor
# -----------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

# -----------------------
# Train/Val split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Preprocess data
X_train_proc = preprocess.fit_transform(X_train)
X_val_proc = preprocess.transform(X_val)

# -----------------------
# Apply SMOTE (numeric + encoded categorical)
# -----------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

# -----------------------
# Models
# -----------------------
# Logistic Regression
pipe_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr.fit(X_train_res, y_train_res)

# Random Forest (optimized)
pipe_rf = RandomForestClassifier(
    n_estimators=50,      # fewer trees
    max_depth=5,          # shallower
    class_weight="balanced",
    random_state=42,
    n_jobs=-1             # use all cores
)
pipe_rf.fit(X_train_res, y_train_res)

# Calibrated Random Forest
rf_base = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
pipe_rf_cal = CalibratedClassifierCV(rf_base, cv=3, method="sigmoid")  # fewer folds
pipe_rf_cal.fit(X_train_res, y_train_res)

# -----------------------
# Compute thresholds
# -----------------------
thr_lr = best_threshold_from_roc(y_val, pipe_lr.predict_proba(X_val_proc)[:, 1])
thr_rf = best_threshold_from_roc(y_val, pipe_rf.predict_proba(X_val_proc)[:, 1])
thr_rf_cal = best_threshold_from_roc(y_val, pipe_rf_cal.predict_proba(X_val_proc)[:, 1])

# -----------------------
# Save models, preprocessor, thresholds
# -----------------------
joblib.dump(preprocess, "preprocess.pkl")
joblib.dump(pipe_lr, "pipe_lr.pkl")
joblib.dump(pipe_rf, "pipe_rf.pkl")
joblib.dump(pipe_rf_cal, "pipe_rf_cal.pkl")
joblib.dump({"lr": thr_lr, "rf": thr_rf, "rf_cal": thr_rf_cal}, "thresholds.pkl")

print("✅ Training complete. Models saved!")
