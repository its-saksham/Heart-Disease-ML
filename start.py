import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

# ===========================
# 1. Load Data
# ===========================
df = pd.read_csv("heart_2022_no_nans.csv")  # update path

print("\n--- Data Overview ---")
print(df.head())
print(df.info())

# ===========================
# 2. Target + Features
# ===========================
target = "HadHeartAttack"
y = df[target].map({"No": 0, "Yes": 1})
X = df.drop(columns=[target])

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(include=np.number).columns

print("\nCategorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numeric_cols))

# ===========================
# 3. Preprocessing
# ===========================
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# First preprocess → get numeric matrix
X_processed = preprocess.fit_transform(X)

# ===========================
# 4. Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# 5. Handle Imbalance with SMOTE
# ===========================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {y_train_res.value_counts().to_dict()}")

# ===========================
# 6. Logistic Regression
# ===========================
log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")
log_reg.fit(X_train_res, y_train_res)
y_pred_lr = log_reg.predict(X_test)

print("\n--- Logistic Regression (SMOTE + Balanced) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ===========================
# 7. Random Forest
# ===========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    class_weight="balanced_subsample",
    random_state=42
)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest (SMOTE + Balanced) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ===========================
# 8. Confusion Matrix
# ===========================
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest (SMOTE + Balanced)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===========================
# 9. ROC Curve
# ===========================
y_prob_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr, tpr, label=f"RF (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (SMOTE + Balanced)")
plt.legend()
plt.show()

# ===========================
# 10. Precision-Recall Curve
# ===========================
prec, rec, _ = precision_recall_curve(y_test, y_prob_rf)
plt.plot(rec, prec, label="Random Forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# ===========================
# 11. Demo Prediction
# ===========================
example = X.iloc[0:1]  # first row
example_processed = preprocess.transform(example)  # preprocess same way
pred = rf.predict(example_processed)[0]
prob = rf.predict_proba(example_processed)[0][1]

print("\n--- Demo Prediction ---")
print("Prediction (0 = No Heart Attack, 1 = Had Heart Attack):", pred)
print(f"Probability of Heart Attack: {prob:.2f}")
