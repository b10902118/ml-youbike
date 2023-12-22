from datetime import datetime, timedelta
import pandas as pd
from test_prophet import Prophet
from sklearn.metrics import mean_absolute_error
from utils import *
from median_optimization import optimal_median

# Define the start and end dates for training and testing
TRAIN_START = "2023-10-02 00:00"
TRAIN_END = "2023-12-17 23:59"
TEST_START = "2023-10-28 00:00"
TEST_END = "2023-10-31 23:59"

# Load data and necessary configurations
with open("./cache/small_data_cache.pkl", "rb") as f:
    df = pd.read_pickle(f)
with open("../html.2023.final.data/sno_test_set.txt") as f:
    ntu_snos = [l.strip() for l in f.read().splitlines()]
with open("./cache/10-03_12_09_rain.pkl", "rb") as f:
    rain_df = pd.read_pickle(f)

# Prepare the data
df['datehour'] = df['time'].dt.floor("H")
rain_df.rename(columns={'datetime': 'datehour'}, inplace=True)
df = df.merge(rain_df, on='datehour', how='left')
df['rain'].fillna(0, inplace=True)

# Prepare the training data for Prophet
train = df[(df['time'] >= TRAIN_START) & (df['time'] <= TRAIN_END)]
train = train.groupby(['time', 'sno']).agg({'sbi': 'mean'}).reset_index()
train.columns = ['ds', 'sno', 'y']

# Prepare the testing data for Prophet
test = df[(df['time'] >= TEST_START) & (df['time'] <= TEST_END)]
test = test.groupby(['time', 'sno']).agg({'sbi': 'mean'}).reset_index()
test.columns = ['ds', 'sno', 'y_true']

# Initialize the result DataFrame
result_df = pd.DataFrame(columns=ntu_snos, dtype=np.float64)

# Define the Prophet model
prophet_model = Prophet()

# Train and predict for each station
for sno, tot in zip(ntu_snos, ntu_tots):
    # Filter data for the current station
    station_train = train[train['sno'] == sno]
    
    # Fit the Prophet model
    prophet_model.fit(station_train[['ds', 'y']])
    
    # Create a DataFrame with future dates for prediction
    future = pd.DataFrame(pd.date_range(TEST_START, TEST_END, freq="20min"), columns=['ds'])
    
    # Make predictions using the Prophet model
    forecast = prophet_model.predict(future)
    
    # Merge the predictions with the result DataFrame
    result_df[sno] = forecast['yhat'].values

# Evaluate the predictions
y_true = test.pivot(index='ds', columns='sno', values='y_true')
mae = mean_absolute_error(y_true.values, result_df.values)
print(f"Mean Absolute Error (MAE): {mae}")
