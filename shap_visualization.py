feature_names = preprocessor_full.get_feature_names_out()
cleaned_feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
X_test_df = pd.DataFrame(X_test, columns=cleaned_feature_names)

# Create SHAP explainer
explainer = shap.Explainer(model, X_test_df)
class_names = ["Low", "Medium", "High"]

# Compute SHAP values
shap_values = explainer(X_test_df, check_additivity=False)

# Summary Plot
shap.summary_plot(shap_values, X_test_df,class_names=class_names, plot_type="bar")