
from asyncore import read
from cgitb import lookup
from email.policy import HTTP
import re
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
from django.shortcuts import render
# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque





import time

import matplotlib.pyplot as plt



import os
import numpy as np
import pandas as pd
import random
LOOKUP_STEP = 15
SCALE = True
N_STEPS = 50

def index (request):
    
    company = request.GET['symbol']
    epochs = int(request.GET['epochs'])
    # skip variable is used to skip the saviing of the newly created model to our model
    if (request.GET['save'] == 'true'):
        save = True
    else:
        save = False
    # use variable is used to use the saved model 
    
    if (request.GET['use'] == 'true'):
        use = True
    else:
        use = False
    
    print(use)
    print(save)
    
    print(request.GET['start'])
    print(request.GET['end'])
    
    # if date is not given, then use today's date
    if (request.GET['start']!=""):
        start = request.GET['start']
    else:
        # get time 6 months ago
        start = (datetime.datetime.now() - datetime.timedelta(days=360)).strftime("%Y-%m-%d")
    
    
    

    if (request.GET['end']!=""):
        end = request.GET['end']
    else:
        #end = time.strftime("%Y-%m-%d") 
        # 10 days ago 
        end = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        
    date_now = time.strftime("%Y-%m-%d")
    
    LOOKUP_STEP = int(request.GET['lookup_step'])
    np.random.seed(314)
    tf.random.set_seed(314)
    random.seed(314)
    scale_str = f"sc-{int(SCALE)}"
    SHUFFLE = True
    shuffle_str = f"sh-{int(SHUFFLE)}"
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    TEST_SIZE = 0.2
    FEATURE_COLUMNS = ["Adj Close", "Volume", "Open", "High", "Low"]
    N_LAYERS = 2
    CELL = LSTM
    UNITS = 256
    DROPOUT = 0.4
    BIDIRECTIONAL = False
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    BATCH_SIZE = 64
    EPOCHS = epochs

    ticker = company
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    
    model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
    {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    
    if BIDIRECTIONAL:
        model_name += "-b"
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("data"):
        os.mkdir("data")

    data = load_data(ticker, start, end, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                    shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                    feature_columns=FEATURE_COLUMNS)

    data["df"].to_csv(ticker_data_filename)
    # construct the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    
    

    if (use):
        checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
        # train the model and save the weights whenever we see 
        # a new optimal model using ModelCheckpoint
        history = model.fit(data["X_train"], data["y_train"],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(data["X_test"], data["y_test"]),
                            callbacks=[checkpointer, tensorboard],
                            verbose=1)
        print(history)
        
    #else: 
    #    history = model.fit(data["X_train"], data["y_train"],
    #                        batch_size=BATCH_SIZE,
    #                        epochs=EPOCHS,
    #                        validation_data=(data["X_test"], data["y_test"]),
    #                        verbose=1)
    #print(history)
    
    

    if (save): 
        model_path = os.path.join("results", model_name) + ".h5"
        model.load_weights(model_path)


    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    
    if SCALE:
        mean_absolute_error = data["column_scaler"]["Adj Close"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae
        
    final_df = get_final_df(model, data)
    future_price = predict(model, data)
    
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    total_buy_profit  = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)
    
    # printing metrics
    print(f"Future price after {LOOKUP_STEP} days: {future_price}")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)
    
   
   
    
    return JsonResponse({'data': f"Future price after {LOOKUP_STEP} days: {future_price}\n" +
                            f"{LOSS} loss: {loss}\n\n" +   
                            f"Mean Absolute Error: {mean_absolute_error}\n" +
                            f"Accuracy score: {accuracy_score}\n\n" +
                            f"Total buy profit: {total_buy_profit}\n\n" +
                            f"Total sell profit: {total_sell_profit}\n\n" +
                            f"Total profit: {total_profit}\n\n" +
                            f"Profit per trade: {profit_per_trade}\n"})
    
                         
        
        
    return HttpResponse(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")

    

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
    
def get_final_df(model, data):

    # if predicted future price is higher than the current, 
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit, 
                                    final_df["Adj Close"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit, 
                                    final_df["Adj Close"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df


def load_data(ticker,start, end, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['Adj Close', 'Volume', 'Open', 'High', 'Low']):

    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = pdr.get_data_yahoo(ticker, start=start, end=end)
        #df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
        print(df)
    elif isinstance(ticker, pd.DataFrame):
        
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Adj Close'].shift(-lookup_step)
 
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop the NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["Adj Close"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


    
def contracts():
    print("Loading contracts...")