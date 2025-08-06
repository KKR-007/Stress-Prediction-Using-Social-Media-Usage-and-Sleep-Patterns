# Imports
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv("stress_prediction_data_merged.csv", low_memory=False)

# 2. Target and Features

y = df["Stress_Level"]
X = df.drop(columns=["Stress_Level"])

# 3. Encode Target Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_  

# 4. Feature Types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_features:
    X[col] = X[col].astype(str)

# 5. Preprocessing 
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Transform Features
X_processed = preprocessor.fit_transform(X)
feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]

# 7. Balance the Dataset with SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_processed, y_encoded)

# 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# 9. Train XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# 10. Feature Importances
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 11. Top 2 Important Features
print("\nTop 2 Important Features:")
print(importance_df.head(2))



