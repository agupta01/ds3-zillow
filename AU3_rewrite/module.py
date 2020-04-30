from scipy.signal import argrelextrema
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hello_world():
    print("hello world")


failed_cities = []


# def areaUnder3Algorithm(data, predictor=ARIMA_50, )

def find_AU3(time_series):
    start = find_start(time_series)
    predicted = ARIMA(time_series, start)
    residuals = calculate_residuals(time_series, predicted)
    end = find_end(residuals)
    return calculate_area(residuals, start, end)


def find_start(series):
    """
    returns recession start date, measured as the largest local maximum ZHVI for a given city
    takes: city (pd.dataframe) [Date, ZHVI_avg_norm]
    returns: start_date (pd.datetime)
    """
    RECESSION_END = '2015'
    VALUE_COL = 'ZHVI_std'
    MAX_THRESHOLD = 1

    # difference, find local maxes, choose appropriate max,

    series_to_recession_end = series[:RECESSION_END]

    recession_maxes = get_local_maxes(series_to_recession_end)

    # Check for presence of local maxes at all
    if len(recession_maxes) == 0:
        last_date = series.idxmax()
        return last_date

    # Find date of recession minimum
    # TODO recession_minimum_date = my_min(city)

    # Filter largest max before min
    recession_start_date = recession_maxes.idxmax()

    return recession_start_date


def find_end(residuals):
    """
    Finds end date by calculating residuals between predicted and actual values, then finding most recent positive residual
    returns recession end date, measured as the first point of intersection between ZHVI and ARIMA_50 for a given city
    takes: city (pd.dataframe) [Date, ZHVI_avg_norm], ARIMA (pd.dataframe) [Date, forecasted_ZHVI_norm] 
    returns: end_date (pd.datetime)
    """
    # Filter only negative residuals representing when actual time-series surpasses predicted
    negative_residuals = residuals[residuals < 0]

   # If time-series surpasses predictions, end date is earliest date surpassed
   # Else, end date last date in time-series
    if (len(negative_residuals) != 0):
        recession_end_date = negative.index[0]
    else:
        recession_end_date = residuals.index[-1]

    return recession_end_date


def calculate_area(residuals, start, end):
    recession_residuals = residuals[start:end]
    return recession_residuals.sum()


# Predictor Functions

def ARIMA(series, start, params=(5, 1, 1)):
    """
    Params:
    city -- time-series dataframe object containing Date and ZHVI columns
    start -- datetime object from index of city representing peak ZHVI
    params -- p, d, and q parameters for ARIMA
    """
    from statsmodels.tsa.arima_model import ARIMA
    START = start

    series_before_recession_start = series[:START]
    dates_after_recession_start = series[START:].index[1:]
    steps = len(dates_after_recession_start)

    # Try to fit arima model, except if no convergence add to list of failed cities
    # TODO use datetime component
    try:
        model = ARIMA(series_before_recession_start, order=(5, 1, 1))
        model_fit = model.fit(disp=0)
        predicted_values = model_fit.forecast(steps)[0]
        return pd.Series(predicted_values, index=dates_after_recession_start)
    except:
        # TODO Handle exception
        pass


def my_min(city_df, threshold=-0.002):
    """
    Finds new 
    Params:
    city_df -- DataFrame with columns ['Date', Metric]
    threshold -- threshold for minimums
    returns: index of minimum value
    """
    THRESHOLD = threshold

    city_df = city_df[(city_df['Date'] > '2010-01-01') &
                      (city_df['Date'] < '2015-01-01')]
    my_zhvi = city_df['ZHVI_std'].to_numpy()

    min_indeces = argrelextrema(my_zhvi, np.less)[0]
    min_vals = [my_zhvi[val] for val in min_indeces]
    min_dates = [city_df['Date'].to_numpy()[val] for val in min_indeces]

    curr_index = 0

    for i in range(len(min_vals)):
        if i == len(min_vals) - 1:
            curr_index = len(min_vals) - 1
            break
        else:
            if np.diff([min_vals[i], min_vals[i + 1]]) <= THRESHOLD:
                curr_index += 1
            else:
                curr_index = i
                break

    return min_dates[curr_index]


def moving_difference(series):
    """
    Differences series to identify extrema.
    """
    MAX_THRESHOLD = 1

    difference_index = series.index[1:]
    difference_values = series[1:].values - series[:-1].values
    differences = pd.Series(difference_index, difference_values)

    # Create filter for maxes where max occurs when difference is positive but difference after is negative
    # TODO Explain MAX_THRESHOLD, implement w/o for loop
    max_filter = np.array([(difference_values[i] >= 0)
                           & (difference_values[i+1] <= 0)
                           & (difference_values[i+1] - difference_values[i] <= MAX_THRESHOLD)
                           for i in range(len(differences) - 1)],
                          dtype=bool)
    max_filter = np.append(max_filter, False)

    maxes = series[1:][max_filter]

    return maxes


def get_local_maxes(series, extrema_function=moving_difference):
    """
    Uses moving difference to identify local maxes
    """
    return moving_difference(series)


def calculate_residuals(actual_series, predicted_series):

    return (predicted_series - actual_series).dropna()
