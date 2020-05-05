"""
Module which provides utilities to analyze time-series depressions, particularly AU3 algorithm specified at [link]
"""

from scipy.signal import argrelextrema
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hello_world():
    print("hello world")


failed_cities = []


class AreaUnderObject():

    #predictors = (ARIMA_predictor)

    def __init__(self, time_series, date_col=None, value_col=None, dates=None, lower_bound=2005, upper_bound=2015, start_date=None, end_date=None, predictor=None):
        """
        Params:
        time-series -- (datetime-indexed series, pd.DataFrame, iterable) array-like iterable with time-series data
            - if pass DataFrame, must specify date and value column
            - if iterable passed, must pass corresponding dates to dates
        start_date -- (str, pd.Datetime) manual specified date for start of depression/AU3 calculation
        end_date -- (str, pd.Datetime) manual specified date for end of depression/AU3 calculation


        """
        if isinstance(lower_bound, int):
            self.__LOWER_BOUND = str(lower_bound)
        else:
            self.__LOWER_BOUND = lower_bound
        if isinstance(upper_bound, int):
            self.__UPPER_BOUND = str(upper_bound)
        else:
            self.__UPPER_BOUND = upper_bound

        self.time_series = time_series

        if predictor == None:
            self.predictor = self.ARIMA_predictor
        else:
            self.predictor = predictor

        # If wait_bool, wait
        # (method to calculate)
        self.result = self.__calculate_AU3()

    def __convert_ts(self):
        pass

    def __calculate_AU3(self):

        start_date = self.__find_start()
        self.start_date = start_date

        predicted = self.predictor()
        self.predicted = predicted

        residuals = self.__calculate_residuals()
        self.residuals = residuals

        end_date = self.__find_end(residuals)
        self.end_date = end_date

        return self.__calculate_area(residuals, start_date, end_date)

    def __find_start(self):
        """
        Method that calculates 1-step difference to get relative extrema, finds local maxes, chooses appropriate max
        Returns: 
        start_date (pd.datetime) -- the identified start date of the depression
        """

        # difference, find local maxes, choose appropriate max,

        series_to_recession_end = self.time_series[:self.__UPPER_BOUND]

        recession_maxes = self.__get_local_maxes(series_to_recession_end)

        # Check for presence of local maxes at all, if not return date with highest value
        if len(recession_maxes) == 0:
            last_date = self.time_series.idxmax()
            return last_date

        # Find date of recession minimum
        # TODO recession_minimum_date = my_min(city)

        recession_start_date = recession_maxes.idxmax()

        return recession_start_date

    def __find_end(self, residuals):
        """
        Finds end date by calculating residuals between predicted and actual values, then finding most recent positive residual
        returns recession end date, measured as the first point of intersection between ZHVI and ARIMA_50 for a given city
        takes: city (pd.dataframe) [Date, ZHVI_avg_norm], ARIMA (pd.dataframe) [Date, forecasted_ZHVI_norm] 
        returns: end_date (pd.datetime)
        """
        residuals = self.residuals

        # Filter only negative residuals representing when actual time-series surpasses predicted
        negative_residuals = residuals[residuals < 0]

    # If time-series surpasses predictions, end date is earliest date surpassed
    # Else, end date last date in time-series
        if (len(negative_residuals) != 0):
            recession_end_date = negative_residuals.index[0]
        else:
            recession_end_date = residuals.index[-1]

        return recession_end_date

    def __calculate_area(self, residuals, start, end):
        recession_residuals = residuals[start:end]
        return recession_residuals.sum()

    # Predictor Functions

    def ARIMA_predictor(self, params=(5, 1, 1)):
        """
        Predictor method based on ARIMA model
        Params:
        params (3- element tuple) -- p, d, and q parameters for ARIMA
        Returns:
        (pd.Datetime-index series) series of indexed values
        """
        from statsmodels.tsa.arima_model import ARIMA
        START_DATE = self.start_date

        series_before_recession_start = self.time_series[:START_DATE]
        dates_after_recession_start = self.time_series[START_DATE:].index[1:]
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

    def __my_min(self, city_df, threshold=-0.002):
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

    def moving_difference(self, series):
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

    def __get_local_maxes(self, series, extrema_function=moving_difference):
        """
        Uses moving difference to identify local maxes
        """
        return self.moving_difference(series)

    def __calculate_residuals(self):

        actual_series = self.time_series
        predicted_series = self.predicted

        return (predicted_series - actual_series).dropna()
