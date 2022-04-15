# used to transform language to readable format for JavaScript and visa versa

from django.contrib.auth.models import User, Group
from rest_framework import serializers
from .models import pricePredict
import math
import numpy as np

import pandas as pd
import matplotlib.pyplot as mpl
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout




class PricePredictionSerializer(serializers.Serializer):
    """
    Serializer for price prediction
    """
    
    symbol = serializers.CharField(max_length=6)
    enddate = serializers.DateField('end date')
    startdate = serializers.DateField('start date')
    
    def machineLearn():
            company = 'AAPL'

            # This will be chosen by the user
            start = dt.datetime(2012, 1, 1)
            end = dt.datetime(2020, 1, 1)

            data = web.DataReader(company, 'yahoo', start, end)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            predictiondays = 60 # this will be chosen by the user


            xtrain = []
            ytrain = []

            for x in range(predictiondays, len(scaled)):
                xtrain.append(scaled[x-predictiondays:x, 0])
                ytrain.append(scaled[x, 0])
                    
            # convert to numpy array

            xtrain, ytrain =  np.array(xtrain), np.array(ytrain)
            xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

            #Building the model 

            # units will be chosen by the user

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50,))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            epochs = 50
            model.fit(xtrain, ytrain, batch_size=32, epochs=epochs)

            # Load Test data 

            teststart = dt.datetime(2020, 1, 1)
            testend = dt.datetime.now()

            testdata = web.DataReader(company, 'yahoo', teststart, testend)
            actualprices = testdata['Close'].values

            totaldataset = pd.concat((data['Close'], testdata['Close']), axis=0)

            model_inputs = totaldataset[len(totaldataset) - len(testdata) - predictiondays:].values
            model_inputs = model_inputs.reshape(-1, 1)
            model_inputs = scaler.transform(model_inputs)

            # Make Prediction 

            xtest = []
            for x in range(predictiondays, len(model_inputs)):
                xtest.append(model_inputs[x-predictiondays:x, 0])

            xtest = np.array(xtest)
            xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)

            predictions = model.predict(xtest)
            predictions = scaler.inverse_transform(predictions)


            mpl.plot(actualprices , color='black', label='Actual')
            mpl.plot(predictions, color='green', label='Predicted')
            mpl.title(f'{company} Stock Price Prediction')
            mpl.xlabel('Time')
            mpl.ylabel(f'{company} Price')
            mpl.legend()
            mpl.show() 

            # Predicting the stock price of MSFT

            print("1")
            realdata = [model_inputs[len(model_inputs) + 1 - predictiondays:len(model_inputs+ 1), 0]]
            realdata = np.array(realdata)
            realdata = realdata.reshape(realdata.shape[0], realdata.shape[1], 1)

            print("2")
            prediction = model.predict(realdata)
            prediction = scaler.inverse_transform(prediction)
            print("3")
            print(f"Prediction: {prediction}")
            
            return prediction


            # Evaluating the model
            
            