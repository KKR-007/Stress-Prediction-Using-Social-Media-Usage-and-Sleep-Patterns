#!pip install ipywidgets shap --quiet

# UI 
import ipywidgets as widgets
from IPython.display import display, clear_output

model_input = widgets.Text(
    value='xgboost',
    placeholder='Enter model name (xgboost / random_forest)',
    description='Model:',
    style={'description_width': 'initial'}
)

run_button = widgets.Button(
    description='Run Model',
    button_style='success',
    tooltip='Train and evaluate model',
    icon='check'
)

output = widgets.Output()

display(model_input, run_button, output)

# TRAINING LOGIC
def run_model_training_manual(button):
    with output:
        clear_output()

        # Required Libraries
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from imblearn.combine import SMOTEENN
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
     

        # Validate Model Name
        MODEL_NAME = model_input.value.strip().lower()
        allowed_models = ["xgboost", "random_forest"]

        if MODEL_NAME not in allowed_models:
            print(f"Invalid model name: '{MODEL_NAME}'. Please use one of: {', '.join(allowed_models)}")
            return

        print(f"Training model: {MODEL_NAME.upper()}")

        # Load Dataset
        df = pd.read_csv("stress_prediction_data_merged.csv")

        # Feature Engineering
        df["Sleep_Deficit"] = df["Sleep_Hours"].apply(lambda x: max(0, 8 - x))
        df["Balanced_Activity_Score"] = df["Physical_Activity_Hours"] - df["Social_Media_Usage"]
        df["Activity_to_Sleep_Ratio"] = df['Physical_Activity_Hours'] / df['Sleep_Hours']
        df["ScreenTime_to_Sleep_Ratio"] = df['Social_Media_Usage'] / df['Sleep_Hours']
        df["Healthy_Lifestyle_Index"] = df['Physical_Activity_Hours'] + df['Sleep_Hours'] - df['Social_Media_Usage']

        # Select Features and Target
        features = [
            "Sleep_Hours", "Social_Media_Usage", "Sleep_Deficit",
            "Activity_to_Sleep_Ratio", "Mental_Health_Condition", "Healthy_Lifestyle_Index",
            "Physical_Activity_Hours", "Balanced_Activity_Score", "ScreenTime_to_Sleep_Ratio", "Age"
        ]
        X = df[features]
        y = df["Stress_Level"]

        # Encode Target Variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Preprocessing (Scaling & Encoding)
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
        ])
        X_processed = preprocessor.fit_transform(X)

        # Handle Imbalanced Data with SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X_processed, y_encoded)

        # Split Data into Train and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
        )

        # Train the Selected Model
        if MODEL_NAME == "xgboost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        elif MODEL_NAME == "random_forest":
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification Report & Accuracy
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        acc = accuracy_score(y_test, y_pred)

        print("\nClassification Report")
        print("Class\tPrecision\tRecall\t\tF1-Score")
        for class_label in le.classes_:
            metrics = report[class_label]
            print(f"{class_label:<7}\t{metrics['precision']:.2f}\t\t{metrics['recall']:.2f}\t\t{metrics['f1-score']:.2f}")
        print(f"\nAccuracy: {acc:.4f}")

        # Confusion Matrix
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=le.classes_, cmap='Blues')
        plt.title(f"Confusion Matrix - {MODEL_NAME.replace('_', ' ').title()}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

# RUN BUTTON 
run_button.on_click(run_model_training_manual)

