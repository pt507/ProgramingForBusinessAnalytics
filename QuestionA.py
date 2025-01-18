import numpy as np
import pandas as pd
from FREDMD_Tools import *

backtest_dates = pd.date_range(start='2005-1-1',end='2011-3-1',freq='M').to_period('M')
horizons=[3,6,9,12]

#add your code here:

# Define a function to load FRED-MD data from file
def load_data(filename):
    """
    This function imports the FRED-MD data from file into a pandas DataFrame.
    It removes the first row of data that contains the transformation values.
    It formats the data index as a monthly Period Index.
    It returns the formatted DataFrame.
    """
    # Read the csv file into a DataFrame, skipping the first row
    df = pd.read_csv(filename, header=0)
    
    # Drop the second row by its index label
    df = df.drop(0)

    # Set the index column to be the first column 'sasdate'
    df = df.set_index('sasdate')
    
    # Convert the index to a Period Index with monthly frequency
    df.index = pd.PeriodIndex(df.index, freq='M')
    return df

# Load and format the '2023-09.csv' FRED-MD file
FRED_DATA = load_data('2023-09.csv')
print(FRED_DATA.head())

# Create empty dataframes to store the results
models = pd.DataFrame(index=FRED_DATA.columns) # to store the chosen model for each series
naive = pd.DataFrame(index=backtest_dates, columns=FRED_DATA.columns) # to store the naive forecasts for each series

for h in horizons:
    # Create dataframes to store the forecast errors for each horizon
    globals()[f'horizon_{h}'] = pd.DataFrame(index=backtest_dates, columns=FRED_DATA.columns)
    # Create dataframes to store the naive forecast errors for each horizon
    globals()[f'naive_{h}'] = pd.DataFrame(index=backtest_dates, columns=FRED_DATA.columns)


# Loop through each backtest date
for date in backtest_dates:
    # Loop through each series in the data
    for series in FRED_DATA.columns:

        # Select the valid data up to the backtest date
        srs = select_continuous(FRED_DATA[series][:date])

        # Fit the models and get the forecasts
        mdl, forecasts = fit_models(srs, h=12)
        # Record the chosen model for the series
        models.loc[series, date] = mdl
        # Record the naive forecast for the series
        naive.loc[date, series] = srs.iloc[-1]

        # Loop through each horizon
        for h in horizons:
            # Select the forecast error for the horizon
            globals()[f'horizon_{h}'].loc[date, series] = (FRED_DATA[series][date+1:date+12] - forecasts)[h-1]
            # Calculate the naive forecast error for the horizon
            globals()[f'naive_{h}'].loc[date, series] = FRED_DATA[series][date+h] - naive.loc[date, series]

# Create empty dataframes to store the R-squared values for each horizon
for h in horizons:
    # Create dataframes with the name R2_hM
    globals()[f'R2_{h}M'] = pd.DataFrame(index=FRED_DATA.columns, columns=['R2'])

# Loop through each series in the data
for series in FRED_DATA.columns:
    # Loop through each horizon
    for h in horizons:
        # Calculate the R-squared value for the horizon
        res = globals()[f'horizon_{h}'].loc[:, series].pow(2).sum()
        tot = globals()[f'naive_{h}'].loc[:, series].pow(2).sum()
        r2 = 1 - res/tot
        # Store the R-squared value in the corresponding dataframe
        globals()[f'R2_{h}M'].at[series, 'R2'] = r2
    
# Create a dataframe with the series identifiers as an index and two columns
fcast_res = pd.DataFrame(index=FRED_DATA.columns, columns=['model', 'R2'])

# Store the latest model selected and the 3 month R2 statistic for each series
fcast_res['model'] = models.iloc[:, -1]
fcast_res['R2'] = R2_3M

# Save the dataframe as a CSV file
fcast_res.to_csv('fcast_res.csv')