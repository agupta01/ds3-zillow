import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import argrelextrema

TEST = 1000


def hello_world():
    print("hello world")

# def plot_my_min(city):
#     city_df = metro_data[metro_data['MetroName'] == city]
#     my_zhvi = city_df['Normalised_ZHVI'].to_numpy()

#     min_indeces = argrelextrema(my_zhvi, np.less, order=1)[0]
#     min_vals = [my_zhvi[val] for val in min_indeces]
#     min_dates = [city_df['Date'].to_numpy()[val] for val in min_indeces]

#     plt.figure(figsize=(6, 3))
#     plt.title('Normalised ZHVI in ' + city)
#     plt.xticks(rotation='vertical')
#     plt.plot(pd.to_datetime(city_df['Date']).dt.date,
#              city_df['Normalised_ZHVI'], color='black')
#     plt.scatter(min_dates, min_vals, color='blue')
#     plt.scatter(min_dates[min_dates.index(my_min(city))],
#                 min_vals[min_dates.index(my_min(city))], color='red')
#     plt.show()


# def my_min(city_df):
#     """
#     """
#     my_zhvi = city_df['Normalised_ZHVI'].to_numpy()

#     min_indeces = argrelextrema(my_zhvi, np.less)[0]
#     min_vals = [my_zhvi[val] for val in min_indeces]
#     min_dates = [city_df['Date'].to_numpy()[val] for val in min_indeces]

#     curr_index = 0

#     for i in range(len(min_vals)):
#         if i == len(min_vals) - 1:
#             curr_index = len(min_vals) - 1
#             break
#         else:
#             if np.diff([min_vals[i], min_vals[i + 1]]) <= THRESHOLD:
#                 curr_index += 1
#             else:
#                 curr_index = i
#                 break

#     return min_dates[curr_index]


# def calc_resid(city, predicted, start, end, residuals=False):
#     """
#     Params:
#     city -- time-series dataframe object containing Date and ZHVI columns
#     predicted -- predicted values from max to last date of city time-series
#     max -- datetime object from index of city representing peak ZHVI
#     end -- datetime object from index of city representing intersection of
#     actual and predicted ZHVI or last date of actual
#     """

#     # get indices of start and end date and use those to splice arrays

#     # get data after start
#     recession_ZHVI = city[city['Date'] >= start]

#     # get data after end [TODO OPTIMIZE]
#     recession_ZHVI = recession_ZHVI[recession_ZHVI['Date'] < end]

#     # get index of recession_ZHVI end, filter predicted array to index, subtract both TODO index together?
#     end_index = len(recession_ZHVI)
#     predicted_to_end = predicted[:end_index]
#     diffs = predicted_to_end - recession_ZHVI['ZHVI_std'].values
#     if residuals:
#         return (np.sum(diffs), diffs)
#     return np.sum(diffs)

#     failed_cities = []


# def ARIMA_50(city, start, params=(5, 1, 1)):
#     """
#     Params:
#     city -- time-series dataframe object containing Date and ZHVI columns
#     start -- datetime object from index of city representing peak ZHVI
#     params -- p, d, and q parameters for ARIMA
#     """

#     # add start.dt.strftime('%Y-%m-%d') to convert datetime to string

#     from statsmodels.tsa.arima_model import ARIMA

#     before = city[['Date', 'ZHVI_std']]

#     before = before[before['Date'] < start].set_index(['Date'])[
#         'ZHVI_std'].values

#     # Number of points for ARIMA to predict is length of original city df minus start indexed df
#     steps = city.shape[0] - before.shape[0]

#     # Try to fit arima model, except if no convergence add to list of failed cities
#     try:
#         model = ARIMA(before, order=(5, 1, 1))
#         model_fit = model.fit(disp=0)
#         return model_fit.forecast(steps)[0]
#     except:
#         failed_cities.append(np.unique(city.CBSA_Code)[0])
#         return np.repeat(city[city['Date'] == start].ZHVI_AllHomes, steps)


# def find_AU3_1(region_data, predictor=ARIMA_50, start=None, end=None, residuals=False, plot=False, time=False):
#     """
#     Takes dataframe with CBSA codes, ZHVI values and name of city
#     start: manual specified start date
#     end: manual specified end date
#     returns: Area between predicted and actual ZHVI values from start to end of recession
#     """
#     # TODO DOCSTRING and doctest?


# #     print("City: {}".format(np.unique(metro.CBSA_Code)))
#     start = my_min(region_data)
#     arima = predictor(region_data, start)
#     print("Start Date: ", start)
# #     fig, ax = plt.subplots()
# #     ax.plot(pd.to_datetime(metro['Date'][4:]).dt.date, movingAverage_normalize(metro['ZHVI_AllHomes'].values))
# #     print(metro[metro['Date'] > start].shape)
#     end = find_end(region_data, start, arima)
#     print("End Date: ", end)
#     return calc_resid(region_data, arima, start, end, residuals=residuals)
