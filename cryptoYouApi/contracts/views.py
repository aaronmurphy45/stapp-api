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


def index(request):
    
    
    return ("Hello World")
    #Predicting the stock price of MSFT
