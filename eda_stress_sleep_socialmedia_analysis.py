# 0. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import chi2

# 1. LOAD DATA
df = pd.read_csv('mental_health_data_final_data.csv')
data = df[['Sleep_Hours', 'Social_Media_Usage', 'Stress_Level']].dropna()

stress_map = {"Low": 0, "Medium": 1, "High": 2}
data['stress_ord'] = data['Stress_Level'].map(stress_map)

# 2. EXPLORATORY DATA VISUALIZATION
sns.set(style="whitegrid")

# Boxplot: Sleep Hours by Stress Level
plt.figure(figsize=(8, 5))
sns.boxplot(x='Stress_Level', y='Sleep_Hours', data=data, palette='coolwarm')
plt.title('Sleep Hours by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Sleep Hours')
plt.tight_layout()
plt.show()

# Boxplot: Social Media Usage by Stress Level
plt.figure(figsize=(8, 5))
sns.boxplot(x='Stress_Level', y='Social_Media_Usage', data=data, palette='coolwarm')
plt.title('Social Media Usage by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Social Media Usage (hours)')
plt.tight_layout()
plt.show()

# Bar plot: Average Social Media Usage by Stress Level
plt.figure(figsize=(8, 6))
sns.barplot(x='Stress_Level', y='Social_Media_Usage', data=data, palette='viridis')
plt.title('Average Social Media Use by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Avg Social Media Usage (hours)')
plt.tight_layout()
plt.show()

# 3. ORDINAL LOGIT MODELING WITH INTERACTION
# Add interaction term
data["interaction"] = data["Sleep_Hours"] * data["Social_Media_Usage"]

# Fit base model without interaction
mod_base = OrderedModel(
    endog=data["stress_ord"],
    exog=data[["Sleep_Hours", "Social_Media_Usage"]],
    distr="logit"
)
res_base = mod_base.fit(method="bfgs", disp=False)

# Fit model with interaction
mod_int = OrderedModel(
    endog=data["stress_ord"],
    exog=data[["Sleep_Hours", "Social_Media_Usage", "interaction"]],
    distr="logit"
)
res_int = mod_int.fit(method="bfgs", disp=False)

# Likelihood Ratio Test
lr_stat = 2 * (res_int.llf - res_base.llf)
p_val = chi2.sf(lr_stat, df=1)
print(f"LR statistic = {lr_stat:,.3f}  |  p = {p_val:.3e}")
if p_val < 0.05:
    print("The interaction term significantly improves model fit.")
else:
    print("he interaction term is NOT significant.")

# 4. VISUALIZE INTERACTION EFFECTS (PROBABILITY OF HIGH STRESS)
sleep_grid = np.linspace(4, 9, 100)
social_grid = np.linspace(0.5, 6, 100)
S, M = np.meshgrid(sleep_grid, social_grid)
grid_df = pd.DataFrame({
    "Sleep_Hours": S.ravel(),
    "Social_Media_Usage": M.ravel(),
    "interaction": (S * M).ravel()
})

probs = res_int.model.predict(res_int.params, exog=grid_df)
high_prob = probs[:, 2].reshape(S.shape)

# Contour plot
fig = plt.figure(figsize=(8, 6))
cs = plt.contourf(S, M, high_prob, levels=20, cmap="RdYlBu_r")
plt.colorbar(cs, label="Predicted P(High Stress)")
plt.xlabel("Sleep Hours")
plt.ylabel("Social Media Usage (hours)")
plt.title("Interaction Effect on Probability of High Stress")
plt.tight_layout()
plt.show()
