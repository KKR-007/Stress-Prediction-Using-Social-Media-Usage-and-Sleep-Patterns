import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
data_path = "stress_prediction_data_merged.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {data_path}")

# 2. Select Required Columns
scaler = StandardScaler()
df[["Sleep_Hours", "Social_Media_Usage", "Physical_Activity_Hours"]] = scaler.fit_transform(
    df[["Sleep_Hours", "Social_Media_Usage", "Physical_Activity_Hours"]]
)
required_cols = ["Sleep_Hours", "Social_Media_Usage", "Physical_Activity_Hours", "Stress_Level"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# 3. Feature Engineering
df["Healthy_Lifestyle_Index"] = (
    df["Physical_Activity_Hours"] + df["Sleep_Hours"] - df["Social_Media_Usage"]
)

# 4. Boxplot: Lifestyle Score vs. Stress Level
plt.figure(figsize=(8, 5))
sns.boxplot(
    x="Stress_Level", y="Healthy_Lifestyle_Index", data=df,
    palette="Set2", showfliers=False
)
plt.title("Healthy Lifestyle Index vs. Stress Level", fontsize=14)
plt.xlabel("Stress Level (0 = Low, 1 = Medium, 2 = High)")
plt.ylabel("Healthy Lifestyle Index")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 5. ANOVA Test
grouped_scores = [group["Healthy_Lifestyle_Index"].values for _, group in df.groupby("Stress_Level")]

# One-way ANOVA
anova_result = f_oneway(*grouped_scores)

# 6. Result
print("\nANOVA Test Result:")
print(f"F-statistic : {anova_result.statistic:.4f}")
print(f"P-value     : {anova_result.pvalue:.4e}")

if anova_result.pvalue < 0.05:
    print("Healthy_Lifestyle_Index significantly differs across stress levels.")
else:
    print("No significant difference in Healthy_Lifestyle_Index across stress levels.")