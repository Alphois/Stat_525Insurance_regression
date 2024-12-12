# Stat_525Insurance_regression
Intermediary versions of data is also present in the github repository as 
"Final_merged_dataset.csv"

"Final_merged_dataset_cleaned.csv"

"Cleaned_Insurance_Data.csv"
"cleaned_data_for_glm.csv"
If you have the above files, then no matter which python script you run, it will generate the output shown in the latex document containing my writeup. 

However, the code will also run with just the RawHistoricalHurricaneDataOne.csv, RawHistoricalHurricaneDataTwo.csv, provided you run it in the sequence shown below:

Run clean_dataset.py to clean and preprocess the data.
Run regression_model.py to generate the Ordinary Least Squares (OLS) model results.
For the Generalized Linear Model (GLM) analysis, run glm_insurance.py. If considering interactions between predictors, run glm_insurance_interaction.py, but know that these two files led to the same result when it came to final model predictors. 

To See the Original Issues experienced (initially 0 was imputed into missing entries, but later the mean of non zero values for two columns was imputed in place of missing values  with the exception of avg_pressure, for which 1013hPa was imputed instead as the average atmospheric pressure at sea level):

Run original\_clean\_dataset.py to see the dataset where 0 imputation was used instead of mean imputation.
Run original\_regression\_model.py to observe the regression model issues caused by 0 imputation (e.g., multicollinearity, non-normality).



Order of Execution:


Start with the dataset cleaning scripts (original\_clean\_dataset.py or clean\_dataset.py).
Proceed to the regression scripts (original\_regression\_model.py or regression\_model.py).
Conclude with GLM scripts (glm\_insurance.py or glm\_insurance\_interaction.py) 
