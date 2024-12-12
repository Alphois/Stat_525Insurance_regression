
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.width', 200)
file_path = 'Final_Merged_Dataset.csv'
data = pd.read_csv(file_path)

# Inspect the data for missing values
missing_values = data.isnull().sum()

# Display the first few rows and missing values summary
data.head(), missing_values

# Impute missing values in hurricane-related variables with 0
data['Avg_Wind_Speed'].fillna(data['Avg_Wind_Speed'].mean(), inplace=True)
data['Avg_Pressure'].fillna(1013, inplace=True)
data['Avg_Dist2Land'].fillna(data['Avg_Dist2Land'].mean(), inplace=True)

# Confirm that there are no missing values left
missing_values_after_imputation = data.isnull().sum()
missing_values_after_imputation

output_cleaned_file_path = 'Cleaned_Insurance_Data.csv'
data.to_csv(output_cleaned_file_path, index=False)
output_cleaned_file_path


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
# vif data showed that avg_wind had a vif score above 5


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


# Create the Premium-to-Insured Value Ratio
data['Premium_Ratio'] = data['Premium'] / data['Total_Insured_Value']

# Define predictors and response
predictors = ['Avg_Dist2Land', 'Avg_Pressure', 'Latitude',
          'Avg_Wind_Speed', 'Longitude', 'Losses', 'Policy_Year']
X = data[predictors]  # Same predictors as before minus the total insured value
y = data['Premium_Ratio']
import seaborn as sns
import matplotlib.pyplot as plt

# Update the list of variables for the correlation matrix
variables_ratio = [
    'Premium_Ratio', 'Avg_Dist2Land', 'Avg_Pressure',
    'Latitude', 'Longitude', 'Losses', 'Policy_Year'
]

# Calculate the correlation matrix
corr_matrix_ratio = data[variables_ratio].corr()

# Print the correlation matrix
print("\nCorrelation Matrix with Premium Ratio:")
print(corr_matrix_ratio)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_ratio, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap with Premium Ratio')
plt.show()
# Define predictors (excluding Total_Insured_Value) and response
predictors_ratio = [
    'Avg_Dist2Land', 'Avg_Pressure',
    'Latitude', 'Longitude', 'Losses', 'Policy_Year', 
]
X_ratio = data[predictors_ratio]
y_ratio = data['Premium_Ratio']

# Add constant for the intercept
X_ratio = add_constant(X_ratio)

# Fit the OLS regression model
model_ratio = OLS(y_ratio, X_ratio).fit()

# Display the regression summary
print(model_ratio.summary())

variables_to_plot = [
    'Avg_Dist2Land', 'Avg_Pressure', 'Latitude', 'Longitude', 'Losses', 'Policy_Year'
]

# Create the pairplot using the specified variables
sns.pairplot(data[variables_to_plot])
plt.show()
# Create a pairplot with 'Premium' as the y-variable

sns.pairplot(data, y_vars='Premium', x_vars=predictors)
plt.show()