# Data in "notebooks" folder

## .ipynb files

### data_preprocessing.ipynb

Preprocess data from Łódź city and splits it for training and updating sets

### LSTM_model_different_window_sizes.ipynb

LSTM model is being learned on different window input size.
Where time distance beetwen each timestamp is one hour.

### LSTM_model_different_data_size.ipynb

LSTM model is being learned on different input data size.
Where time distance beetwen each timestamp is one hour.

### LSTM_model_for_each_30_min.ipynb

LSTM model for time distance beetwen each timestamp == half of hour.
Where input data is being shuffle and not. Comparison.
Better predictions gives model learned with "shuffle data mode".

Here both models are being saved to pickle files with biases for each model inside.

The models with shuffle mode - because of not learning module in pipeline - are taken as an input to the pipelnie backend.

Mean and standard deviation of the data is also calculated here and saved to the pickle file and taken as an input to the pipeline backend module for transforming data when predictions are being made.

### update_LSTM_model.ipynb

File contains updating of LSTM model on new data. Here are patches from 28 days.
Then the comparison with basic model - without updates - is being made.

### for_another_cities_predictions.ipynb

Comparison of predictions for different polish cities. LSTM model is being learned on only from Łódź city data.
Comparisons are being made with bias inclusion for LSTM and xGBoost models.

### linear_regression.ipynb

Old notebook - it was used to generate results which compare LSTM with naive method and linear regression. Results are being saved in MSE_error_for_lstm_and_naive_method_comparison.txt file.

## folders

### data_distance_from_Lodz

Folder with .csv files for making comparisons beetwen predictions for each city. - it needs to be updated with new folders structure!

### data_for_main_model

Data for training and updating from Łódź.

### lstm_models

.pkl files with lstm models - there is no data - will be added in the nearest future to generate automatically!

### xgb_models

.pkl files with xgb models - there is no data - will be added in the nearest future to generate automatically!

## .py file

### different_functions.py

A script with common functions for each notebook
