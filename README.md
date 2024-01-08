# Data in "notebooks" folder

## Jupyter Notebook files

### Notebook 0: data_preprocessing.ipynb

Preprocesses data from Łódź city and splits it for training, validation and testing sets.

### Notebook 1: LSTM_model_different_windows_sizes_and_50epochs.ipynb

LSTM models fitting and evaluation.

Here three models are being saved to pikle files and taken as an input to the pipelnie backend.

Min and max values of the data is also calculated here and saved to the pickle file and taken as an input to the pipeline backend module for transforming data when predictions are being made.

### Notebook 2: LSTM_model_update.ipynb

File contains updating of LSTM model with new data.

### Notebook 3: XGB_model.ipynb

Training of each model on previously prepared data and biases calculations.

Here three models with proper biases are being saved to pikle files.

### Notebook 4: naive_method_and_linear_regression.ipynb

Generates results for comparison of LSTM models with naive method and linear regression.

### Notebook 5: other_cities_predictions.ipynb

Comparison of predictions for different polish cities. LSTM model is being learned on only from Łódź city data.
Comparisons are being made with bias inclusion xGBoost models.

Attention: file depends on lstm_models and xgb_models folders. First execute XGB_model.ipynb and LSTM_model_different_windows_sizes_and_50epochs.ipynb notebooks.

## Python file

### different_functions.py

A script with common functions for each notebook.

## Folders

### all_data/data_distance_from_Lodz

Folder with .csv files for making comparisons beetwen predictions for different cities.

### all_data/data_for_main_model

Data from Łódź city for training and updating LSTM model.

### all_data/data_for_XGB

Combined data from Warszawa, Wrocław, Szczecin, Rzeszów for XGBoost model.

### generated_models/lstm_models

.pkl files with lstm models - there is no data - first execute LSTM_model_different_windows_sizes_and_50epochs.ipynb notebook.

### generated_models/xgb_models

.pkl files with xgb models - there is no data - first execute XGB_model.ipynb notebook.
