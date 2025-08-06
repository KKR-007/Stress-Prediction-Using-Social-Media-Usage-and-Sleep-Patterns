# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

# 1. Load Dataset
df = pd.read_csv("stress_prediction_data_merged.csv")

# 2. Feature Engineering
df["Sleep_Deficit"] = df["Sleep_Hours"].apply(lambda x: max(0, 8 - x))
df["Balanced_Activity_Score"] = df["Physical_Activity_Hours"] - df["Social_Media_Usage"]
df["Activity_to_Sleep_Ratio"] = df["Physical_Activity_Hours"] / df["Sleep_Hours"]
df["ScreenTime_to_Sleep_Ratio"] = df["Social_Media_Usage"] / df["Sleep_Hours"]
df["Healthy_Lifestyle_Index"] = df["Physical_Activity_Hours"] + df["Sleep_Hours"] - df["Social_Media_Usage"]

# 3. Encode Target Labels
le = LabelEncoder()
df["Stress_Level_Encoded"] = le.fit_transform(df["Stress_Level"])


# 4. Basic Feature Model
features_basic = [
    "Physical_Activity_Hours", "Mental_Health_Condition",
    "Gender", "Age", "Sleep_Hours", "Social_Media_Usage"
]
X_basic = df[features_basic]
y_basic = df["Stress_Level_Encoded"]

numeric_basic = X_basic.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_basic = X_basic.select_dtypes(include=["object", "category"]).columns.tolist()

# 5. Preprocessing for Basic Features
preprocessor_basic = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_basic),
    ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_basic)
])

X_basic_processed = preprocessor_basic.fit_transform(X_basic)

# 6. Balance Dataset with SMOTEENN (Basic)
Xb_resampled, yb_resampled = SMOTEENN(random_state=42).fit_resample(X_basic_processed, y_basic)

# 7. Train/Test Split (Basic Model)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb_resampled, yb_resampled, test_size=0.2, stratify=yb_resampled, random_state=42
)

# 8. Train XGBoost Classifier (Basic)
model_basic = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_basic.fit(Xb_train, yb_train)
yb_pred = model_basic.predict(Xb_test)
acc_basic = accuracy_score(yb_test, yb_pred)
print(f"Accuracy using basic features: {acc_basic:.4f}")

# 9. Full Feature Model
features_full = [
    "Sleep_Hours", "Social_Media_Usage", "Sleep_Deficit",
    "Activity_to_Sleep_Ratio", "Mental_Health_Condition", "Healthy_Lifestyle_Index",
    "Physical_Activity_Hours", "Balanced_Activity_Score", "ScreenTime_to_Sleep_Ratio", "Age"
]
X_full = df[features_full]
y_full = df["Stress_Level_Encoded"]

numeric_full = X_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_full = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

# 10. Preprocessing for Full Features
preprocessor_full = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_full),
    ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_full)
])

X_full_processed = preprocessor_full.fit_transform(X_full)

# 11. Balance Dataset with SMOTEENN (Full)
Xf_resampled, yf_resampled = SMOTEENN(random_state=42).fit_resample(X_full_processed, y_full)

# 12. Train/Test Split (Full Model)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    Xf_resampled, yf_resampled, test_size=0.2, stratify=yf_resampled, random_state=42
)

# 13. Train XGBoost Classifier (Full)
model_full = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_full.fit(Xf_train, yf_train)
yf_pred = model_full.predict(Xf_test)
acc_full = accuracy_score(yf_test, yf_pred)
print(f"Accuracy using full engineered features: {acc_full:.4f}")

# 14. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    model_full, Xf_test, yf_test, display_labels=le.classes_, cmap='Blues'
)
plt.title("Confusion Matrix - Full Model")
plt.grid(False)
plt.tight_layout()
plt.show()

# 15. Accuracy Comparison - Basic vs Full
accuracy_data = pd.DataFrame({
    "Model": ["Basic Features", "Full Engineered Features"],
    "Accuracy": [acc_basic, acc_full]
})

plt.figure(figsize=(7, 5))
sns.barplot(data=accuracy_data, x="Model", y="Accuracy", palette="Set2")
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.6)

for index, row in accuracy_data.iterrows():
    plt.text(index, row.Accuracy + 0.01, f"{row.Accuracy:.2%}", ha='center', fontsize=11)

plt.tight_layout()
plt.show()
