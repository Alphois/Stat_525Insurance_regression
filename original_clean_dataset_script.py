import pandas as pd

# Load the uploaded CSV files
file_one_path = 'RawHistoricalHurricaneDataOne.csv'
file_two_path = 'RawHistoricalHurricaneDataTwo.csv'


data_one = pd.read_csv(file_one_path)
data_two = pd.read_csv(file_two_path)

# Display the initial rows and structure of both datasets to understand their content
data_one_head = data_one.head()
data_two_head = data_two.head()

data_one_info = data_one.info()
data_two_info = data_two.info()

data_one_head, data_one_info, data_two_head, data_two_info

# Rename columns to give meaningful names based on context
data_one.columns = ['ID', 'Season', 'Number', 'Basin', 'SubBasin', 'Name', 'ISO_Time',
                    'Nature', 'Latitude', 'Longitude', 'Wind_Speed', 'Pressure',
                    'Agency', 'Track_Type', 'Dist2Land', 'Extra1', 'Extra2']
data_two.columns = ['Storm_Name', 'Date', 'Longitude', 'Latitude', 'Quadrant',
                    'Wind_Speed', 'Wind_Radius', 'Extra1', 'Extra2', 'Extra3', 'Extra4']

# Drop completely null columns
data_one = data_one.drop(columns=['Extra1', 'Extra2'])
data_two = data_two.drop(columns=['Extra1', 'Extra2', 'Extra3', 'Extra4'])

# Standardize key columns for merging (e.g., Latitude, Longitude, Wind_Speed)
data_one['Latitude'] = pd.to_numeric(data_one['Latitude'], errors='coerce')
data_one['Longitude'] = pd.to_numeric(data_one['Longitude'], errors='coerce')
data_one['Wind_Speed'] = pd.to_numeric(data_one['Wind_Speed'], errors='coerce')

data_two['Latitude'] = pd.to_numeric(data_two['Latitude'], errors='coerce')
data_two['Longitude'] = pd.to_numeric(data_two['Longitude'], errors='coerce')
data_two['Wind_Speed'] = pd.to_numeric(data_two['Wind_Speed'], errors='coerce')

# Handle missing hurricane names in the first dataset by using nearby matching entries
data_one['Name'] = data_one['Name'].fillna(method='ffill')

# Merge datasets on approximate matches of key attributes: Latitude, Longitude, Wind_Speed, and Date/Time
merged_data = pd.merge(
    data_one,
    data_two,
    left_on=['Latitude', 'Longitude', 'Wind_Speed'],
    right_on=['Latitude', 'Longitude', 'Wind_Speed'],
    how='outer',
    suffixes=('_One', '_Two')
)

# # Display the cleaned and merged dataset
# import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned and Merged Hurricane Data", dataframe=merged_data)
merged_data.to_csv('Cleaned_Hurricane_Data.csv', index=False)
print("Data has been saved to Cleaned_Hurricane_Data.csv")


# cleaned hurricane data with code above
# # Display the cleaned and merged dataset
# import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned and Merged Hurricane Data", dataframe=merged_data)

# Clean Insurance Premium Dataset
# Dropping irrelevant rows and renaming columns based on initial inspection
import pandas as pd

# Load the datasets
insurance_premium_data = pd.read_csv('Insurance_premiums.csv')
hurricane_data = pd.read_csv('Cleaned_Hurricane_Data.csv')

# Display the first few rows of each dataset to understand their structure
insurance_premium_data.head(), hurricane_data.head()
insurance_premium_data_cleaned = insurance_premium_data.iloc[4:].rename(columns={
    'Unnamed: 1': 'Latitude',
    'Unnamed: 2': 'Longitude',
    'Unnamed: 3': 'Total_Insured_Value',
    'Unnamed: 4': 'Premium',
    'Unnamed: 5': 'Losses',
    'Unnamed: 6': 'Policy_Year'
})

# Selecting only relevant columns
insurance_premium_data_cleaned = insurance_premium_data_cleaned[['Latitude', 'Longitude', 'Total_Insured_Value', 'Premium', 'Losses', 'Policy_Year']]
insurance_premium_data_cleaned.dropna(how='all', inplace=True)

# Convert numerical fields to appropriate types and clean currency/number formatting
insurance_premium_data_cleaned['Total_Insured_Value'] = insurance_premium_data_cleaned['Total_Insured_Value'].replace(',', '', regex=True).astype(float)
insurance_premium_data_cleaned['Premium'] = insurance_premium_data_cleaned['Premium'].replace(',', '', regex=True).astype(float)
insurance_premium_data_cleaned['Losses'] = insurance_premium_data_cleaned['Losses'].replace(',', '', regex=True).astype(float)
insurance_premium_data_cleaned['Policy_Year'] = insurance_premium_data_cleaned['Policy_Year'].astype(int)

# Clean Hurricane Dataset
# Selecting relevant columns
hurricane_data_cleaned = hurricane_data[['Season', 'Latitude', 'Longitude', 'Wind_Speed', 'Pressure', 'Dist2Land']]

# Drop rows with missing values in key columns
hurricane_data_cleaned.dropna(subset=['Latitude', 'Longitude', 'Wind_Speed', 'Pressure', 'Dist2Land'], inplace=True)

# Display cleaned dataframes to confirm structure
hurricane_data_cleaned.to_csv('Cleaned_Hurricane_Data.csv')
insurance_premium_data_cleaned.to_csv('Cleaned_insurance_premiums')
# insurance premium data
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Load the datasets
insurance_premium_data = pd.read_csv('Insurance_premiums.csv')
hurricane_data = pd.read_csv('Cleaned_Hurricane_Data.csv')

# **Data Cleaning and Preprocessing**

## Clean Insurance Premium Dataset
# Skip the first few irrelevant rows and rename columns
insurance_premium_data_cleaned = insurance_premium_data.iloc[4:].rename(columns={
    'Unnamed: 1': 'Latitude',
    'Unnamed: 2': 'Longitude',
    'Unnamed: 3': 'Total_Insured_Value',
    'Unnamed: 4': 'Premium',
    'Unnamed: 5': 'Losses',
    'Unnamed: 6': 'Policy_Year'
})

# Select relevant columns
insurance_premium_data_cleaned = insurance_premium_data_cleaned[['Latitude', 'Longitude', 'Total_Insured_Value', 'Premium', 'Losses', 'Policy_Year']]
insurance_premium_data_cleaned.dropna(how='all', inplace=True)

# Convert columns to appropriate data types and clean formatting
insurance_premium_data_cleaned['Latitude'] = pd.to_numeric(insurance_premium_data_cleaned['Latitude'], errors='coerce')
insurance_premium_data_cleaned['Longitude'] = pd.to_numeric(insurance_premium_data_cleaned['Longitude'], errors='coerce')
insurance_premium_data_cleaned['Total_Insured_Value'] = insurance_premium_data_cleaned['Total_Insured_Value'].replace(',', '', regex=True).astype(float)
insurance_premium_data_cleaned['Premium'] = insurance_premium_data_cleaned['Premium'].replace(',', '', regex=True).astype(float)
insurance_premium_data_cleaned['Losses'] = insurance_premium_data_cleaned['Losses'].replace(',', '', regex=True).astype(float)
insurance_premium_data_cleaned['Policy_Year'] = insurance_premium_data_cleaned['Policy_Year'].astype(int)

# Drop rows with missing or invalid latitude/longitude values
insurance_premium_data_cleaned.dropna(subset=['Latitude', 'Longitude'], inplace=True)




## Clean Hurricane Dataset
# Select relevant columns

hurricane_data_cleaned = hurricane_data[['Season', 'Latitude', 'Longitude', 'Wind_Speed', 'Pressure', 'Dist2Land']]
# Convert columns to appropriate data types
hurricane_data_cleaned['Latitude'] = pd.to_numeric(hurricane_data_cleaned['Latitude'], errors='coerce')
hurricane_data_cleaned['Longitude'] = pd.to_numeric(hurricane_data_cleaned['Longitude'], errors='coerce')
hurricane_data_cleaned['Wind_Speed'] = pd.to_numeric(hurricane_data_cleaned['Wind_Speed'], errors='coerce')
hurricane_data_cleaned['Pressure'] = pd.to_numeric(hurricane_data_cleaned['Pressure'], errors='coerce')
hurricane_data_cleaned['Dist2Land'] = pd.to_numeric(hurricane_data_cleaned['Dist2Land'], errors='coerce')

# Drop rows with missing or invalid data in key columns
hurricane_data_cleaned.dropna(subset=['Latitude', 'Longitude', 'Wind_Speed', 'Pressure', 'Dist2Land'], inplace=True)

# Reset index after cleaning
hurricane_data_cleaned.reset_index(drop=True, inplace=True)

# **Data Merging and Aggregation**

## Reset indices for insurance dataset
insurance_premium_data_cleaned.reset_index(drop=True, inplace=True)

## Convert latitude and longitude to radians for spatial calculations
insurance_coords = np.radians(insurance_premium_data_cleaned[['Latitude', 'Longitude']])
hurricane_coords = np.radians(hurricane_data_cleaned[['Latitude', 'Longitude']])

## Create a k-d tree for hurricane locations
hurricane_tree = cKDTree(hurricane_coords)

## Define radius threshold for hurricane impact (in miles and radians)
radius_threshold = 100  # miles
radius_radians = radius_threshold / 3958.8  # Earth's radius in miles

## Query the k-d tree to find hurricanes within the radius for each insurance location
indices = hurricane_tree.query_ball_point(insurance_coords, r=radius_radians)

## Aggregate hurricane data for each insurance location
aggregated_features = []

for idx, nearby_indices in enumerate(indices):
    if nearby_indices:
        nearby_hurricanes = hurricane_data_cleaned.iloc[nearby_indices]
        aggregated_features.append({
            'Latitude': insurance_premium_data_cleaned.iloc[idx]['Latitude'],
            'Longitude': insurance_premium_data_cleaned.iloc[idx]['Longitude'],
            'Avg_Wind_Speed': nearby_hurricanes['Wind_Speed'].mean(),
            'Avg_Pressure': nearby_hurricanes['Pressure'].mean(),
            'Avg_Dist2Land': nearby_hurricanes['Dist2Land'].mean()
        })
    else:
        aggregated_features.append({
            'Latitude': insurance_premium_data_cleaned.iloc[idx]['Latitude'],
            'Longitude': insurance_premium_data_cleaned.iloc[idx]['Longitude'],
            'Avg_Wind_Speed': np.nan,
            'Avg_Pressure': np.nan,
            'Avg_Dist2Land': np.nan
        })

## Convert the aggregated features into a DataFrame
aggregated_hurricane_data_final = pd.DataFrame(aggregated_features)

## Merge the aggregated hurricane data with the insurance premium dataset
final_merged_data = pd.merge(
    insurance_premium_data_cleaned,
    aggregated_hurricane_data_final,
    on=['Latitude', 'Longitude'],
    how='left'
)
final_merged_data.drop_duplicates(inplace=True)
# replace zero values with avg atmospheric pressure in hPa at sea level

# **Final Merged Dataset**
# Display or save the final merged dataset
print(final_merged_data.head())

# Optionally, save to a CSV file
final_merged_data.to_csv('Final_Merged_Dataset.csv', index=False)


# input 0s for missing avg windspeed avg pressure and avg_dist2land columns

# Impute missing values in hurricane-related variables with 0
# Impute missing values in hurricane-related variables with 0
final_merged_data['Avg_Wind_Speed'].fillna(0, inplace=True)
final_merged_data['Avg_Pressure'].fillna(0, inplace=True)
final_merged_data['Avg_Dist2Land'].fillna(0, inplace=True)

# Confirm that there are no missing values left
missing_values_after_imputation = final_merged_data.isnull().sum()
print(missing_values_after_imputation)

final_merged_data = final_merged_data.groupby(
    ['Latitude', 'Longitude', 'Policy_Year'], as_index=False
).agg({
    'Avg_Wind_Speed': 'mean',
    'Avg_Pressure': 'mean',
    'Avg_Dist2Land': 'mean',
    'Total_Insured_Value': 'sum',
    'Premium': 'sum',
    'Losses': 'sum'
})

final_merged_data.to_csv('Final_Merged_Dataset_Cleaned.csv', index=False)
print("Cleaned dataset saved successfully.")

# missing_values_after_imputation
# running regression

