from asyncore import read
from email.policy import HTTP
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as mpl
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout


import pandas_datareader as web
import datetime as dt
import mplfinance as mpf

def index(request):
    
    company = request.GET['symbol']
    epochs = int(request.GET['epochs'])
    #start = dt.datetime(request.GET['start'])
    #end = dt.datetime(request.GET['end'])
    
    if 'start' in request.GET:
        start = request.GET['start']
    else:
        start = dt.datetime(2018, 1, 1)
        
        
    if 'end' in request.GET:
        end = request.GET['end']
    else:
        end = dt.datetime.now()
    
    
    #start =request.GET.get('start')
    #end = request.GET.get('end')

    # This will be chosen by the user
    

    data = web.DataReader(company, 'yahoo', start, end)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    predictiondays = 1 # this will be chosen by the user
    #longitude = self.request.query_params.get('longitude')
    #latitude= self.request.query_params.get('latitude')
    #radius = self.request.query_params.get('radius')


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
    #epochs = 10
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
    for x in range(predictiondays, len(model_inputs)+1):
        xtest.append(model_inputs[x-predictiondays:x, 0])

    xtest = np.array(xtest)
    xtest = np.reshape(xtest,(xtest.shape[0], xtest.shape[1], 1))

    predictions = model.predict(xtest)
    predictions = scaler.inverse_transform(predictions)
  
   

    mpl.plot(actualprices , color='black', label='Actual')
    mpl.plot(predictions, color='green', label='Predicted')
    mpl.title(f'{company} Stock Price Prediction')
    mpl.xlabel('Time')
    mpl.ylabel(f'{company} Price')
    mpl.legend()
    #mpl.show() 
    #mpl.savefig('static/images/prediction.png')
    

    print("1")
    print(len(model_inputs))
    realdata = [model_inputs[len(model_inputs) + 1 - predictiondays:len(model_inputs+ 1), 0]]
    print("2")
    
   
   
    realdata = np.array(realdata)
    print("3")
    realdata = np.reshape(realdata, (realdata.shape[0], realdata.shape[1], 1))
    print("4")
    
   
    # numpy reshape realdata to be (None, 1, 1)
    #realdata = np.reshape(realdata, (realdata.shape[0], realdata.shape[1], 1))
    print("5")
    
    
    
    # set real data to keras model shape (None, 1, 1)
    
    np.array(realdata)
    
    realdata = np.reshape(realdata, (realdata.shape[0], realdata.shape[1], 1))
    
    
    
    
    
    
    
    #model.add(tf.keras.layers.Dense(256, input_shape=((None, 1, 1),), activation='sigmoid'))

    
    prediction = model.predict(realdata)
    print("3")
    prediction = scaler.inverse_transform(prediction)
    print("3")
    print(f"Prediction: {prediction}")
    
    return HttpResponse(f"Prediction: {prediction}")

    #Predicting the stock price of MSFT
