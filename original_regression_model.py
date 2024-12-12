import pandas as pd
pd.set_option('display.width', 200)
file_path = 'Final_Merged_Dataset.csv'
data = pd.read_csv(file_path)

# Inspect the data for missing values
missing_values = data.isnull().sum()

# Display the first few rows and missing values summary
data.head(), missing_values

# Impute missing values in hurricane-related variables with 0
data['Avg_Wind_Speed'].fillna(0, inplace=True)
data['Avg_Pressure'].fillna(0, inplace=True)
data['Avg_Dist2Land'].fillna(0, inplace=True)

# Confirm that there are no missing values left
missing_values_after_imputation = data.isnull().sum()
missing_values_after_imputation

output_cleaned_file_path = 'Cleaned_Insurance_Data.csv'
data.to_csv(output_cleaned_file_path, index=False)
output_cleaned_file_path

import statsmodels.api as sm

# Define predictors (X) and response variable (y)
X = data[['Total_Insured_Value', 'Avg_Dist2Land', 'Avg_Pressure', 'Latitude',
          'Avg_Wind_Speed', 'Longitude', 'Losses', 'Policy_Year']]
y = data['Premium']

# Add a constant to the predictors for the regression model
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the regression results
model_summary = model.summary()
with open("Regression_Summary.txt", "w") as file:
    file.write(model.summary().as_text())

# Display the regression summary explicitly
print(model.summary())




# deal with multicollinearity issues
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
# vif data showed that avg_wind had a vif score close to 10. 


# Drop Avg_Wind_Speed from predictors
X_dropped = X.drop(columns=["Avg_Wind_Speed"])

# Fit the regression model with updated predictors
model_updated = sm.OLS(y, X_dropped).fit()

# Display the updated regression results
print(model_updated.summary())
with open("Regression_post_drop.txt", "w") as file:
    file.write(model_updated.summary().as_text())

# check variance inflation factors a second time to see if there is still multicollinearity issues
# Calculate Variance Inflation Factor (VIF) for each predictor
vif_data = pd.DataFrame()
vif_data['Variable'] = X_dropped.columns
vif_data['VIF'] = [variance_inflation_factor(X_dropped.values, i) for i in range(X_dropped.shape[1])]
print(vif_data)

import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan

# Calculate residuals
residuals = model_updated.resid
fitted_values = model_updated.fittedvalues

# Plot histogram of residuals for normality
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q Plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

# Residuals vs. Fitted Values plot for constant variance
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

import scipy.stats as stats

# Shapiro-Wilk Test for normality
shapiro_test_stat, shapiro_p_value = stats.shapiro(model_updated.resid)
print("Shapiro-Wilk Test:")
print(f"Test Statistic: {shapiro_test_stat}")
print(f"p-value: {shapiro_p_value}")

# Jarque-Bera Test for normality
jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(model_updated.resid)
print("\nJarque-Bera Test:")
print(f"Test Statistic: {jarque_bera_stat}")
print(f"p-value: {jarque_bera_p_value}")


from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Calculate residuals and fitted values
residuals = model_updated.resid
fitted_values = model_updated.fittedvalues

# Breusch-Pagan Test - constant variance assumption checking
bp_test_stat, bp_p_value, bp_f_stat, bp_f_p_value = het_breuschpagan(residuals, X)
print("Breusch-Pagan Test:")
print(f"Lagrange Multiplier Statistic: {bp_test_stat}")
print(f"p-value: {bp_p_value}")
print(f"F-statistic: {bp_f_stat}")
print(f"F-test p-value: {bp_f_p_value}")


from scipy.stats import boxcox
import numpy as np


# Define the predictors explicitly
predictors = [
    'Total_Insured_Value', 'Avg_Dist2Land', 'Avg_Pressure',
    'Latitude', 'Longitude', 'Losses', 'Policy_Year'
]

# Check if the response variable (Premium) is positive
if (data['Premium'] <= 0).any():
    data['Premium'] += abs(data['Premium'].min()) + 1  # Shift to positive values

# Apply the Box-Cox transformation
data['Premium_BoxCox'], lambda_boxcox = boxcox(data['Premium'])

# Print the optimal lambda
print(f"Optimal Lambda for Box-Cox Transformation: {lambda_boxcox}")

# Define predictors and response
X = data[predictors]
y = data['Premium_BoxCox']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the regression model with transformed response
model_boxcox = sm.OLS(y, X).fit()

# Display the regression summary
print(model_boxcox.summary())

# redo tests for constant variance and normality to see if box cox worked
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro, jarque_bera

# Residuals and fitted values from the Box-Cox transformed model
residuals = model_boxcox.resid
fitted_values = model_boxcox.fittedvalues

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Histogram of Residuals (Box-Cox Transformed)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q Plot
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Box-Cox Transformed)')
plt.show()

# Residuals vs. Fitted Values plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values (Box-Cox Transformed)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Shapiro-Wilk Test
shapiro_test_stat, shapiro_p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}")

# Jarque-Bera Test
jarque_bera_stat, jarque_bera_p_value = jarque_bera(residuals)
print(f"Jarque-Bera Test Statistic: {jarque_bera_stat}, p-value: {jarque_bera_p_value}")

# Breusch-Pagan Test for constant variance
bp_test_stat, bp_p_value, _, _ = het_breuschpagan(residuals, model_boxcox.model.exog)
print(f"Breusch-Pagan Test Statistic: {bp_test_stat}, p-value: {bp_p_value}")



# consider removing outliers to deal with normality assumption violation
data['Standardized_Residuals'] = model_boxcox.get_influence().resid_studentized_internal

# Define a threshold for identifying outliers (e.g., |standardized residual| > 3)
outlier_threshold = 3
outliers = data[np.abs(data['Standardized_Residuals']) > outlier_threshold]

# Print the number of outliers identified
print(f"Number of outliers identified: {len(outliers)}")

# Remove outliers
data_cleaned = data[np.abs(data['Standardized_Residuals']) <= outlier_threshold]

# Refit the model without outliers
X_cleaned = data_cleaned[predictors]
y_cleaned = data_cleaned['Premium_BoxCox']
X_cleaned = sm.add_constant(X_cleaned)

# Fit the OLS model with cleaned data
model_cleaned = sm.OLS(y_cleaned, X_cleaned).fit()

# Display the regression summary
print(model_cleaned.summary())

# Recheck residual diagnostics
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Plot Q-Q Plot for cleaned model
plt.figure(figsize=(8, 6))
probplot(model_cleaned.resid, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals (Cleaned Data)")
plt.show()


# removing outliers did not seem to help, consider different regression approach - possibly LASSO
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Prepare predictors and response
X = data_cleaned[predictors]
y = data_cleaned['Premium_BoxCox']

# Lasso requires standardized predictors
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardizes the data
    ('lasso', LassoCV(cv=5, random_state=42))  # Lasso with cross-validation
])

# Fit the Lasso model
lasso_pipeline.fit(X, y)

# Get the coefficients and best alpha (regularization strength)
lasso_coefficients = lasso_pipeline.named_steps['lasso'].coef_
lasso_best_alpha = lasso_pipeline.named_steps['lasso'].alpha_
print(f"Best Regularization Strength (Alpha): {lasso_best_alpha}")
print(f"Lasso Coefficients: {lasso_coefficients}")

# Identify the selected predictors
selected_predictors = [pred for pred, coef in zip(predictors, lasso_coefficients) if coef != 0]
print(f"Selected Predictors: {selected_predictors}")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'data_cleaned' is your DataFrame after any previous cleaning steps

# Select relevant variables
variables = [
    'Premium', 'Total_Insured_Value', 'Avg_Dist2Land', 'Avg_Pressure',
    'Latitude', 'Longitude', 'Losses', 'Policy_Year'
]

# Calculate the covariance matrix
cov_matrix = data_cleaned[variables].cov()
print("Covariance Matrix:")
print(cov_matrix)

# Calculate the correlation matrix
corr_matrix = data_cleaned[variables].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Create the Premium-to-Insured Value Ratio
data_cleaned['Premium_Ratio'] = data_cleaned['Premium'] / data_cleaned['Total_Insured_Value']

# Define predictors and response
X = data_cleaned[predictors]  # Same predictors as before
y = data_cleaned['Premium_Ratio']
import seaborn as sns
import matplotlib.pyplot as plt

# Update the list of variables for the correlation matrix
variables_ratio = [
    'Premium_Ratio', 'Avg_Dist2Land', 'Avg_Pressure',
    'Latitude', 'Longitude', 'Losses', 'Policy_Year'
]

# Calculate the correlation matrix
corr_matrix_ratio = data_cleaned[variables_ratio].corr()

# Print the correlation matrix
print("\nCorrelation Matrix with Premium Ratio:")
print(corr_matrix_ratio)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_ratio, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap with Premium Ratio')
plt.show()


from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define predictors (excluding Total_Insured_Value) and response
predictors_ratio = [
    'Avg_Dist2Land', 'Avg_Pressure',
    'Latitude', 'Longitude', 'Losses', 'Policy_Year'
]
X_ratio = data_cleaned[predictors_ratio]
y_ratio = data_cleaned['Premium_Ratio']

# Add constant for the intercept
X_ratio = add_constant(X_ratio)

# Fit the OLS regression model
model_ratio = OLS(y_ratio, X_ratio).fit()

# Display the regression summary
print(model_ratio.summary())

# Check multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = X_ratio.columns
vif_data['VIF'] = [variance_inflation_factor(X_ratio.values, i) for i in range(X_ratio.shape[1])]

# Display VIF results
print("\nVariance Inflation Factors (VIF):")
print(vif_data)



import matplotlib.pyplot as plt
from scipy.stats import probplot, shapiro, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

# Residuals and fitted values
residuals = model_ratio.resid
fitted_values = model_ratio.fittedvalues

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot
plt.figure(figsize=(8, 6))
probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

# Residuals vs. fitted values plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk Test: Statistic = {shapiro_stat}, p-value = {shapiro_p}")

# Jarque-Bera Test
jb_stat, jb_p = jarque_bera(residuals)
print(f"Jarque-Bera Test: Statistic = {jb_stat}, p-value = {jb_p}")

# Breusch-Pagan Test
bp_stat, bp_p_value, _, _ = het_breuschpagan(residuals, X_ratio)
print(f"Breusch-Pagan Test: Statistic = {bp_stat}, p-value = {bp_p_value}")
from scipy.stats import boxcox
import numpy as np

# Ensure the response variable is positive
if (data_cleaned['Premium_Ratio'] <= 0).any():
    data_cleaned['Premium_Ratio'] += abs(data_cleaned['Premium_Ratio'].min()) + 1

# Apply Box-Cox transformation
data_cleaned['Premium_Ratio_BoxCox'], lambda_boxcox = boxcox(data_cleaned['Premium_Ratio'])

# Print the optimal lambda
print(f"Optimal Lambda for Box-Cox Transformation: {lambda_boxcox}")

# Refit the OLS model with the Box-Cox transformed response
X_ratio_boxcox = data_cleaned[predictors_ratio]
y_ratio_boxcox = data_cleaned['Premium_Ratio_BoxCox']
X_ratio_boxcox = sm.add_constant(X_ratio_boxcox)

model_ratio_boxcox = OLS(y_ratio_boxcox, X_ratio_boxcox).fit()

# Display the regression summary
print(model_ratio_boxcox.summary())

import matplotlib.pyplot as plt
from scipy.stats import probplot, shapiro, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

# Residuals and fitted values from the Box-Cox transformed model
residuals_boxcox = model_ratio_boxcox.resid
fitted_values_boxcox = model_ratio_boxcox.fittedvalues

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals_boxcox, bins=30, edgecolor='k')
plt.title('Histogram of Residuals (Box-Cox Transformed)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q Plot
plt.figure(figsize=(8, 6))
probplot(residuals_boxcox, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Box-Cox Transformed)')
plt.show()

# Residuals vs. Fitted Values plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values_boxcox, residuals_boxcox, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values (Box-Cox Transformed)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals_boxcox)
print(f"Shapiro-Wilk Test: Statistic = {shapiro_stat}, p-value = {shapiro_p}")

# Jarque-Bera Test
jb_stat, jb_p = jarque_bera(residuals_boxcox)
print(f"Jarque-Bera Test: Statistic = {jb_stat}, p-value = {jb_p}")

# Breusch-Pagan Test
bp_stat, bp_p_value, _, _ = het_breuschpagan(residuals_boxcox, model_ratio_boxcox.model.exog)
print(f"Breusch-Pagan Test: Statistic = {bp_stat}, p-value = {bp_p_value}")

# model assumptions appear to not be satisfied, so move onto trying generalized linear model (chapter 14.14 in applied linear statistical models textbook)
# before creating new script, save current dataset to new csv so it is accessible to other script
# create new python script for generalized linear models
# Save the cleaned data to a CSV file
data_cleaned.to_csv('cleaned_data_for_glm.csv', index=False)

