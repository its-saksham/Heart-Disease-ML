# train.py 
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

def best_threshold_from_roc(y_true, y_scores):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr
    ix = np.argmax(j)
    th = thr[ix]
    if not np.isfinite(th):
        th = 0.5
    return float(th)

df = pd.read_csv("heart_2022_no_nans.csv")
y = df["HadHeartAttack"].map({"Yes": 1, "No": 0})
X = df.drop("HadHeartAttack", axis=1)

num_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "HeightInMeters", "WeightInKilograms", "BMI"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_proc = preprocess.fit_transform(X_train)
X_val_proc = preprocess.transform(X_val)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

pipe_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr.fit(X_train_res, y_train_res)

pipe_rf = RandomForestClassifier(
    n_estimators=50,      
    max_depth=5,          
    class_weight="balanced",
    random_state=42,
    n_jobs=-1             
)
pipe_rf.fit(X_train_res, y_train_res)

rf_base = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
pipe_rf_cal = CalibratedClassifierCV(rf_base, cv=3, method="sigmoid")  
pipe_rf_cal.fit(X_train_res, y_train_res)

thr_lr = best_threshold_from_roc(y_val, pipe_lr.predict_proba(X_val_proc)[:, 1])
thr_rf = best_threshold_from_roc(y_val, pipe_rf.predict_proba(X_val_proc)[:, 1])
thr_rf_cal = best_threshold_from_roc(y_val, pipe_rf_cal.predict_proba(X_val_proc)[:, 1])

joblib.dump(preprocess, "preprocess.pkl")
joblib.dump(pipe_lr, "pipe_lr.pkl")
joblib.dump(pipe_rf, "pipe_rf.pkl")
joblib.dump(pipe_rf_cal, "pipe_rf_cal.pkl")
joblib.dump({"lr": thr_lr, "rf": thr_rf, "rf_cal": thr_rf_cal}, "thresholds.pkl")

print("✅ Training complete. Models saved!")

