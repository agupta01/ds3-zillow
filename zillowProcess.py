import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

city = pd.read_csv("City_time_series.csv")
prefilter_shape = city.shape[0]

# Begin filtering
# create subtable with just 3bedroom prices, cityID, and Date
# add new column for true if a 3bedroom price exists for that row (use isnull)
# group each cityId by number of true values (see last dsc10 lecture for how to do this)
cols = ["Date", "RegionName", "MedianListingPricePerSqft_3Bedroom"]
city_filter1 = city.filter(cols)
city_filter1 = city_filter1[city_filter1["MedianListingPricePerSqft_3Bedroom"].isnull() == False]
cityList = np.unique(city_filter1['RegionName'].values)
postfilter_shape = city_filter1.shape[0]

# Get cities with most data on 3bedroom prices
city_filter1 = city_filter1.groupby("RegionName").count()
city_filter1['Region'] = cityList
city_filter1 = city_filter1.sort_values("MedianListingPricePerSqft_3Bedroom", ascending=False)

print(city_filter1.head())
print(city_filter1.shape)
print("Filtered", (prefilter_shape - postfilter_shape), "values")

leading_cities = city_filter1[city_filter1['MedianListingPricePerSqft_3Bedroom'] == 96]
print(leading_cities.shape)
