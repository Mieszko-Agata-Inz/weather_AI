from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import numpy as np

###################################################################################################################
########### Script with functions common for all LSTM model notebooks located in root/notebooks folder ############
###################################################################################################################


####################### LSTM models ########################


# function to train lstm model for predictions for first hour
# returns trained model
#
# input params:
# X_train : training input set
# y_train : training set, true values
# X_val : validation input set
# y_val : validation set, true values
# epochs=10 : number of epochs for model
# learning_rate=0.001 : number of learning rate
# batch_size=32 : size of batch
# shuffle=False : should be data shuffled
# window_size=12 : number of timestamps in the window
# weather_components_size=3 : how many weather components to predict
#
# output params:
# model : trained model
# history : history of training part
def LSTM_model1(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    shuffle=False,
    window_size=12,
    weather_components_size=3,
):
    # model creation
    model = Sequential()
    model.add(
        LSTM(
            units=54,
            batch_input_shape=(batch_size, window_size, weather_components_size),
            stateful=True,
            return_sequences=False,
        )
    )
    model.add(Dense(weather_components_size))

    # model compile with given values
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )

    # model training with given training and valid data
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=0,
    )

    return model, history


# function to train lstm model for predictions for second hour
# returns trained model
#
# input params:
# X_train : training input set
# y_train : training set, true values
# X_val : validation input set
# y_val : validation set, true values
# epochs=10 : number of epochs for model
# learning_rate=0.001 : number of learning rate
# batch_size=64 : size of batch
# shuffle=False : should be data shuffled
# window_size=12 : number of timestamps in the window
# weather_components_size=3 : how many weather components to predict
#
# output params:
# model : trained model
# history : history of training part
def LSTM_model2(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    shuffle=False,
    window_size=12,
    weather_components_size=3,
):
    # model creation
    model = Sequential()
    model.add(
        LSTM(
            units=54,
            batch_input_shape=(batch_size, window_size, weather_components_size),
            stateful=True,
            return_sequences=False,
        )
    )
    model.add(Dropout(0.1))
    model.add(Dense(weather_components_size))

    # model compile with given values
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )

    # model training with given training and valid data
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=0,
    )

    return model, history


# function to train lstm model for predictions for third hour
# returns trained model
#
# input params:
# X_train : training input set
# y_train : training set, true values
# X_val : validation input set
# y_val : validation set, true values
# epochs=10 : number of epochs for model
# learning_rate=0.001 : number of learning rate
# batch_size=64 : size of batch
# shuffle=False : should be data shuffled
# window_size=12 : number of timestamps in the window
# weather_components_size=3 : how many weather components to predict
#
# output params:
# model : trained model
# history : history of training part
def LSTM_model3(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    shuffle=False,
    window_size=12,
    weather_components_size=3,
):
    # model creation
    model = Sequential()
    model.add(
        LSTM(
            units=40,
            batch_input_shape=(batch_size, window_size, weather_components_size),
            stateful=True,
            return_sequences=False,
        )
    )
    model.add(Dense(3))

    # model compile with given values
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )

    # model training with given training and valid data
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=0,
    )

    return model, history


####################### additional functions ########################


# function for min max normalization
#
# input params:
# X : input set for making predictions
# y : exact values to compare with predictions
# index : index of column to normalize data in
# min : minimal values of data training set
# max : maximal values of data training set
#
# output params:
# X : modificated input set for making predictions
# y : modificated exact values to compare with predictions
def min_max_normalize(X, y, index, max, min):
    X[:, :, index] = (X[:, :, index] - min) / (max - min)
    y[:, index] = (y[:, index] - min) / (max - min)
    return X, y


# function for min max denormalization
#
# input params:
# y : values of predictions
# min : minimal values of data training set
# max : maximal values of data training set
#
# output params:
# y : modificated values of predictions
def min_max_denormalization(y, max, min):
    y = y * (max - min) + min
    return y


# function for transform data based on window size for model input data and timestamp which to predict
# data is being normalized with min_max_normalize function, which is important to notice
# returns explanatory X and response y variables
#
# input params:
# df : dataframe with data to transform
# max=0 : maximal values of data training set
# min=0 : minimal values of data training set
# window_size=12 : number of timestamps in the window
# timestamps_count=0,
# is_update=False : bool value is it data for update the LSTM models
# for_linear_regression=False : bool value if is it data for linear regression algorithm
#
# output params:
# X : set for making predictions
# y : exact values to compare with predictions
def transform_data(
    df,
    max=0,
    min=0,
    window_size=12,
    timestamps_count=0,
    is_update=False,
    for_linear_regression=False,
):
    if is_update == False:
        df = df.to_numpy()
    data_count = len(df)
    X = []
    y = []

    # dataframe df division into explanatory X and response y variables
    for i in range(data_count - window_size - timestamps_count):
        row = [r for r in df[i : i + window_size]]
        X.append(row)
        label = [df[i + window_size + timestamps_count]]
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    if for_linear_regression == False:
        # reshape y
        resh0 = y.shape[0]
        resh1 = y.shape[2]
        y = y.reshape(resh0, resh1)

        # data normalizations
        weather_components_size = df.shape[1]
        for i in range(weather_components_size):
            X, y = min_max_normalize(X, y, i, max[i], min[i])

    return X, y


# function makes predictions
# predictions are being denormalized
# returns predicted and actual values
#
# input params:
# model
# X : input set for making predictions on
# y : exact values to compare with predictions
# min : minimal values of data training set
# max : maximal values of data training set
#
# output params:
# pred : predicted values
# actual : exact values
def make_predictions(model, X, y, min, max):
    predictions = model.predict(
        X,
        verbose=0,
        batch_size=32,
    )
    weather_components_size = y.shape[1]
    pred = []
    actual = []

    for i in range(weather_components_size):
        pred.append(min_max_denormalization(predictions[:, i], max[i], min[i]))
        actual.append(min_max_denormalization(y[:, i], max[i], min[i]))

    return pred, actual
