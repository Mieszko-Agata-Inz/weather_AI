###################################################################################################################
############## Script with functions common for all in root/notebooks folder LSTM model notebooks #################
###################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model


################### standard_score
# function for standard score calculations
def normalize(X, y, index, training_mean, training_std, timestamps_count=0):
    X[:, :, index] = (X[:, :, index] - training_mean) / training_std
    y[:, index] = (y[:, index] - training_mean) / training_std


################### transform_and_split_data
# split of data into training, validation and test data sets for lstm model,
# also data mean and standard  deviation for data normalization
def transform_and_split_data(
    df_input, window_size=6, test_size=0.1, valid_size=0.1, timestamps_count=0
):
    df = df_input.to_numpy()
    features_len = df_input.shape[1]
    data_count = len(df)
    X = []
    y = []
    # Explanatory X and Response y variables
    for i in range(data_count - window_size - timestamps_count):
        row = [r for r in df[i : i + window_size]]
        X.append(row)
        label = [df[i + window_size + timestamps_count]]
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    resh0 = y.shape[0]
    resh1 = y.shape[2]
    y = y.reshape(resh0, resh1)
    # data split
    train_size = 1 - test_size - valid_size
    train_last_el = int(data_count * train_size)
    valid_last_el = train_last_el + int(data_count * valid_size)

    X_train, y_train = X[:train_last_el], y[:train_last_el]

    X_val, y_val = X[train_last_el:valid_last_el], y[train_last_el:valid_last_el]
    X_test, y_test = X[valid_last_el:], y[valid_last_el:]

    mean = []
    std = []
    # data normalization
    for i in range(features_len):
        mean.append(np.mean(X[:, :, i]))
        std.append(np.std(X[:, :, i]))
        normalize(X_train, y_train, i, mean[i], std[i])
        normalize(X_val, y_val, i, mean[i], std[i])
        normalize(X_test, y_test, i, mean[i], std[i])

    return X_train, y_train, X_val, y_val, X_test, y_test, mean, std


################### transform_and_split_data_update
# split of data into training, validation and test data sets for lstm model,
# also data mean and standard  deviation for data normalization
def transform_and_split_data_update(
    df_input, training_mean, training_std, window_size=12, timestamps_count=0
):
    # df = df_input.to_numpy()
    features_len = df_input.shape[1]
    data_count = len(df_input)
    X = []
    y = []
    # Explanatory X and Response y variables
    for i in range(data_count - window_size - timestamps_count):
        row = [r for r in df_input[i : i + window_size]]
        X.append(row)
        label = [df_input[i + window_size + timestamps_count]]
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    resh0 = y.shape[0]
    resh1 = y.shape[2]
    y = y.reshape(resh0, resh1)

    # data normalization
    for i in range(features_len):
        normalize(X, y, i, training_mean[i], training_std[i])

    return X, y


################### LSTM_model
# train lstm model
def LSTM_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    learning_rate=0.001,
    window_size=6,
    batch_size=32,
    shuffle=False,
):
    model = Sequential()
    model.add(InputLayer((window_size, 3)))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, "ReLU"))  # previous was ReLU
    model.add(Dense(3, "linear"))
    # model = Sequential()
    # model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 3)))
    # model.add(LSTM(16, 'ReLU')) # previous was ReLU
    # model.add(Dense(32, 'ReLU')) # previous was ReLU
    # model.add(Dense(3, 'linear'))

    # cp = ModelCheckpoint('LSTM_model', save_best_only=True)
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
    )  # , callbacks=[cp])
    return model


################### denormalization
# denormalization of data
def denormalization(y, mean, std):
    y = (y * std) + mean
    return y


################### make_predictions
# making predictions
def make_predictions(model, X, y, mean, std):
    predictions = model.predict(X)

    features_len = y.shape[1]
    pred = []
    actual = []

    for i in range(features_len):
        pred.append(denormalization(predictions[:, i], mean[i], std[i]))
        actual.append(denormalization(y[:, i], mean[i], std[i]))

    return pred, actual


################### plot_predictions_and_actual
# denormalization of data
def plot_predictions_and_actual(pred, actual, start=24, end=48):
    fig, axs = plt.subplots(4)
    fig.suptitle(
        "Weather parameters - red means predicted - x: number of input data , y: predicted value"
    )
    fig.tight_layout(pad=1.8)
    axs[0].plot(pred[0][start:end], "r")
    axs[0].plot(actual[0][start:end])
    axs[0].set_title("relative humidity [%]")

    axs[1].plot(pred[1][start:end], "r")
    axs[1].plot(actual[1][start:end])
    axs[1].set_title("speed of wind [km/h]")

    axs[2].plot(pred[2][start:end], "r")
    axs[2].plot(actual[2][start:end])
    axs[2].set_title("temperature")

    axs[3].plot(actual[2][:])
    axs[3].set_title("temperature [st C]")
