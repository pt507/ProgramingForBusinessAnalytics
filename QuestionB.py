import numpy as np
import pandas as pd
from FREDMD_Tools import *

backtest_dates = pd.date_range(start='2005-1-1',end='2021-3-1',freq='M').to_period('M')

#add your code here:

#Import the files ’2023-09-TF.csv’ and ’NBER_DATES.csv’
FRED_TF = pd.read_csv('2023-09-TF.csv', header=0)
NBER_DATES = pd.read_csv('NBER_DATES.csv')

# format the indices appropriately
# Set the index column to be the first column 'sasdate'
FRED_TF = FRED_TF.set_index('sasdate')
NBER_DATES = NBER_DATES.set_index('Unnamed: 0')
# Convert the index to a Period Index with monthly frequency
FRED_TF.index = pd.PeriodIndex(FRED_TF.index, freq='M')
NBER_DATES.index = pd.PeriodIndex(NBER_DATES.index, freq='M')

# convert the recession column to binary values (1 for recession, 0 for expansion)
NBER_DATES['0'] = NBER_DATES['0'].map({'Recession': 1, 'Expansion': 0})

# Set up a dataframe ’prob results’ to hold the probabilities of recession obtained for the models
prob_results = pd.DataFrame(index=backtest_dates, columns=['LogisticRegression', 'SVC'])

# Define a function to perform the backtesting steps
def perform_backtest(backtest_date, FRED_TF, NBER_DATES):
    # Select the full set of time series data available at time t
    selected_data = FRED_TF.loc[:backtest_date]
    
    # Remove time series with less than 36 non-missing values
    selected_data = selected_data.dropna(thresh=36, axis=1)
    
    # Standardize each series and fill missing values with zeros
    selected_data = selected_data.apply(lambda x: (x - x.mean()) / x.std()).fillna(0)
    
    # Compute 8 Principal Components
    principal_components = pca_function(selected_data, 8)

    # Shift the recession indicator data back 6 months
    shifted_recession_data = NBER_DATES.shift(-6)

    # Select appropriate values for each month in Dt
    Dt_values = shifted_recession_data.loc['1959-01':backtest_date].values
    
    # Use fit_class_models to fit classification models
    lr_predict, svc_predict, _, _ = fit_class_models(principal_components, Dt_values)
    
    # Store the probabilities of recession in 'prob_results'
    prob_results.loc[backtest_date] = [lr_predict[0, 1], svc_predict[0, 1]]

# Loop through each backtest date
for backtest_date in backtest_dates:
    perform_backtest(backtest_date, FRED_TF, NBER_DATES)

print(prob_results)
# Create variables to store Brier Scores
brier_lr = 0
brier_svc = 0
brier_comb = 0

# Create a column in prob_results to store Combination model probabilities
prob_results['Comb'] = 0.5 * (prob_results['LogisticRegression'] + prob_results['SVC'])

# Shift NBER data back 6 months
shifted_NBER_data = NBER_DATES.shift(-6)

# Loop through each backtest date
for backtest_date in backtest_dates:
    # Get the actual NBER classification for the current backtest date
    actual_classification = shifted_NBER_data.loc[backtest_date].values
    
    # Get the predicted probabilities for each model
    lr_prob = prob_results.loc[backtest_date, 'LogisticRegression']
    svc_prob = prob_results.loc[backtest_date, 'SVC']
    comb_prob = prob_results.loc[backtest_date, 'Comb']
    
    # Calculate Brier Scores for each model
    brier_lr += (lr_prob - actual_classification) ** 2
    brier_svc += (svc_prob - actual_classification) ** 2
    brier_comb += (comb_prob - actual_classification) ** 2

# Divide by the number of backtest dates to get the average Brier Scores
brier_lr /= len(backtest_dates)
brier_svc /= len(backtest_dates)
brier_comb /= len(backtest_dates)

# Display the Brier Scores
print("Brier Score for Logistic Regression model:", brier_lr)
print("Brier Score for SVC model:", brier_svc)
print("Brier Score for Combination model:", brier_comb)


# Create a dictionary to store Brier Scores for each model
brier_scores = {
    'LogisticRegression': brier_lr,
    'SVC': brier_svc,
    'Comb': brier_comb
}

# Find the model with the lowest Brier Score
best_model = min(brier_scores, key=brier_scores.get)
best_brier_score = brier_scores[best_model]

# Display the best model and its Brier Score
print("Best Predictive Model:", best_model)
print("Brier Score for the Best Model:", best_brier_score)

# Select the corresponding recession probability values for the best model
best_model_probabilities = prob_results[best_model]

# Write the selected values to 'RecessionIndicator.csv'
best_model_probabilities.to_csv('RecessionIndicator.csv', index=True)