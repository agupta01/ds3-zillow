from scipy.signal import argrelextrema
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hello_world():
    print("hello world")


# def areaUnder3Algorithm(data, predictor=ARIMA_50, )


def find_start(city):
    """
    returns recession start date, measured as the largest local maximum ZHVI for a given city
    takes: city (pd.dataframe) [Date, ZHVI_avg_norm]
    returns: start_date (pd.datetime)
    """
    RECESSION_END = '2015'
    VALUE_COL = 'ZHVI_std'
    THRESHOLD = 1

    # Get latest date from city
    last_date = city.sort_values('Date', ascending=False).iloc[0]['Date']
    city = city[city['Date'] < RECESSION_END]

    def moving_difference(index):
        return city[VALUE_COL].iloc[index] - city[VALUE_COL].iloc[index-1]

    diffs = np.array([moving_difference(i) for i in range(1, len(city))])

    # Reshape dataframe to include diffs, then add diffs
    city = city.iloc[1:]
    city['Diffs'] = diffs

    # Find local maxes using diffs
    is_max = np.array([(city['Diffs'].iloc[i] >= 0)
                       and (city['Diffs'].iloc[i+1] <= 0)
                       and (city['Diffs'].iloc[i+1] - city['Diffs'].iloc[i] <= THRESHOLD) for i in range(len(city) - 1)])
    is_max = np.append(is_max, False)

    # Check for presence of local maxes at all
    if np.count_nonzero(is_max) == 0:
        return last_date

    # Add 'is_maximum' truth column to dataframe
    city['Max'] = is_max

    # Find date of recession minimum
    recession_minimum_date = my_min(city1)

    # Filter and find largest max
    recession_max = city[(city['Max'] == 1.0)
                         & (city['Date'] < recession_minimum_date)].sort_values("ZHVI_std", ascending=False).iloc[0]

    start_date = recession_max['Date']

    return start_date


def my_min(city_df, thershold=-0.002):
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
