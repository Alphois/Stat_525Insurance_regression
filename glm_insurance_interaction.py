import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.genmod.families import Gamma

# ---------------------------------------------------
# Load and prepare data
# ---------------------------------------------------
df = pd.read_csv("cleaned_data_for_glm.csv")

# Confirm that the necessary columns are in the dataset
required_columns = [
    'Premium_Ratio',
    'Avg_Dist2Land',
    'Avg_Pressure',
    'Latitude',
    'Longitude',
    'Losses',
    'Policy_Year'
]
# Optional predictor if decide to keep avg_wind_speed:
# 'Avg_Wind_Speed'

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# If you want to include the optional predictor, ensure it exists:
if 'Avg_Wind_Speed' in df.columns:
    candidate_predictors = required_columns[1:] + ['Avg_Wind_Speed']
    print("avg_wind speed added")
else: 
    candidate_predictors = required_columns[1:]

candidate_predictors = [
    'Avg_Dist2Land',
    'Avg_Pressure',
    'Latitude',
    'Longitude',
    'Losses',
    'Policy_Year'
]

# If you have the optional predictor and want to include it:
# candidate_predictors.append('Avg_Wind_Speed')

# ---------------------------------------------------
# Define a function to fit a Gamma GLM with a log link
# ---------------------------------------------------
def fit_model(formula, data):
    """Fit a Gamma GLM with log link given a formula."""
    model = smf.glm(formula=formula, data=data, family=sm.families.Gamma(sm.families.links.log()))
    result = model.fit()
    return result

# ---------------------------------------------------
# Forward Selection
# ---------------------------------------------------
current_predictors = []
current_formula = "Premium_Ratio ~ 1"
current_model = fit_model(current_formula, df)

from scipy.stats import chi2

# ... (other code remains the same)

while True:
    improved = False
    best_pval = 1.0
    best_predictor = None
    best_model = None

    for pred in candidate_predictors:
        if pred not in current_predictors:
            # Try adding this predictor
            trial_predictors = current_predictors + [pred]
            if pred == 'Longitude' and 'Losses' in current_predictors:
                trial_predictors.append('Longitude:Losses')
            trial_formula = "Premium_Ratio ~ " + " + ".join(trial_predictors)
            trial_model = fit_model(trial_formula, df)

            # Likelihood Ratio Test
            lr_stat = 2 * (trial_model.llf - current_model.llf)
            lr_df = current_model.df_resid - trial_model.df_resid

            # Compute p-value using scipy.stats chi2
            p_value = 1 - chi2.cdf(lr_stat, lr_df)

            if (p_value < 0.05) and (p_value < best_pval):
                best_pval = p_value
                best_predictor = pred
                best_model = trial_model

    if best_predictor is not None:
        current_predictors.append(best_predictor)
        current_model = best_model
        print(f"Added {best_predictor} with p-value {best_pval:.5f}")
        improved = True
    else:
        break

# ---------------------------------------------------
# Final Model Results
# ---------------------------------------------------
final_model = current_model
print("\nFinal Model Summary:")
print(final_model.summary())

# ---------------------------------------------------
# Diagnostic Plots
# ---------------------------------------------------
residuals = final_model.resid_deviance
fitted_values = final_model.fittedvalues

# 1. Residuals vs. Fitted Values
plt.figure(figsize=(6,4))
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Deviance Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()

# 2. Histogram of residuals
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# 3. Q-Q Plot of residuals
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()

# 4. Residuals vs. each predictor
for pred in current_predictors:
    plt.figure(figsize=(6,4))
    plt.scatter(df[pred], residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(pred)
    plt.ylabel("Deviance Residuals")
    plt.title(f"Residuals vs. {pred}")
    plt.show()


# Assuming `df` is your DataFrame containing numerical predictors and the response variable Premium_Ratio
# Adjust 'vars' to include the variables you are interested in
vars_to_plot = ['Premium_Ratio', 'Avg_Dist2Land', 'Avg_Pressure', 'Latitude', 'Longitude', 'Losses', 'Policy_Year']

sns.pairplot(df[vars_to_plot], diag_kind='hist', corner=True)
plt.tight_layout()
plt.show()
