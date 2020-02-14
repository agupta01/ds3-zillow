#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:14:23 2020

@author: Luke
"""
import pandas as pd
import numpy as np
import statsmodels.tsa.arima_model.ARIMA

def calc_resid(city, predicted, max, end):
    """
    Params:
    city -- time-series dataframe object containing Date and ZHVI columns
    predicted -- predicted values from max to last date of city time-series
    max -- datetime object from index of city representing peak ZHVI
    end -- datetime object from index of city representing intersection of 
    actual and predicted ZHVI or last date of actual
    """
    actual = city[city['Date'] > max and city['Date'] < end]['ZHVI']
    predicted_np = predicted[:actual.size]
    actual_np = actual.to_numpy(dtype=np.float)
    
    return np.subtract(predicted, actual_np)


def ARIMA_50(city, max):
    """
    Params:
    city -- time-series dataframe object containing Date and ZHVI columns
    max -- datetime object from index of city representing peak ZHVI
    """
    before = city[city['Date'] < max]
    model = ARIMA(before, (5, 1, 1))
    
    return model.predict(start, city['Date'].iloc[-1])

    