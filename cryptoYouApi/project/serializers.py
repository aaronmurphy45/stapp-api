# used to transform language to readable format for JavaScript and visa versa

from django.contrib.auth.models import User, Group
from rest_framework import serializers
from .models import pricePredict, Contracts
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
        
        print("hello")
        

class ContractsSerializer(serializers.Serializer):
    """
    Serializer for contracts
    """
    
    symbol = serializers.CharField(max_length=6)
    enddate = serializers.DateField('end date')
    startdate = serializers.DateField('start date')
    
    def machineLearn():
        
        print("hello")