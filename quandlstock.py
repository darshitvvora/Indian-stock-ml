
import quandl
quandl.ApiConfig.api_key = '4D8hkYAV4WEkcTmD9LMW'

import pandas as pd

# add quandl API key for unrestricted


# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call
data = quandl.get_table('WIKI/PRICES', ticker = ['AAPL', 'MSFT', 'WMT'],
                        qopts = { 'columns': ['ticker', 'date', 'adj_close'] },
                        date = { 'gte': '2015-12-31', 'lte': '2016-12-31' },
                        paginate=True)

# create a new dataframe with 'date' column as index
new = data.set_index('date')

# use pandas pivot function to sort adj_close by tickers
clean_data = new.pivot(columns='ticker')