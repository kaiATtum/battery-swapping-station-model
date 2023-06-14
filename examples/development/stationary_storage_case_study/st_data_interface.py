import numpy as np
import os
import pandas as pd
import mesmo
import datetime

class data_stationary_storage(object):

    wholesale_price_data: pd.DataFrame
    battery_data: pd.DataFrame

    def __init__(
            self,
            data_path: str,
    ):
        mesmo.utils.logger.info('loading dataset...')

        self.wholesale_price_data_half_hour = pd.read_excel(os.path.join(data_path, 'market_data.xlsx'),'Half-hourly data')
        self.wholesale_price_data_daily = pd.read_excel(os.path.join(data_path, 'market_data.xlsx'), 'Daily data')
        self.battery_data = pd.read_excel(os.path.join(data_path, 'battery_data.xlsx'))
        self.battery_data.rename({'Unnamed: 0': 'parameter'}, axis=1, inplace=True)

        self.time_stamp_half_hour = self.wholesale_price_data_half_hour['Unnamed: 0'].values
        self.time_stamp_daily = self.wholesale_price_data_daily['Unnamed: 0'].values

        mesmo.utils.logger.info('Warning: It seems there is duplicated values in the time stamp data.')

        # list of duplicated half-hour time stamps:
        x = pd.Series(self.wholesale_price_data_half_hour['Unnamed: 0'].values)
        # x[x.duplicated()]
        # 3988 2018 - 03 - 25 T02: 00:00.000000000
        # 3989 2018 - 03 - 25 T02: 30:00.000000000
        # 21796 2019 - 03 - 31T02: 00:00.000000000
        # 21797 2019 - 03 - 31T02: 30:00.000000000
        # 39268 2020 - 03 - 29 T02: 00:00.000000000
        # 39269 2020 - 03 - 29 T02: 30:00.000000000
        print("duplicated time stamps")
        print(x[x.duplicated()])

        # manually fixing data
        self.time_stamp_half_hour[3986] = np.datetime64('2018-03-25T01:00:00.000000000')
        self.time_stamp_half_hour[3987] = np.datetime64('2018-03-25T01:30:00.000000000')
        self.time_stamp_half_hour[21794] = np.datetime64('2019-03-31T01:00:00.000000000')
        self.time_stamp_half_hour[21795] = np.datetime64('2019-03-31T01:30:00.000000000')
        self.time_stamp_half_hour[39266] = np.datetime64('2020-03-29T01:00:00.000000000')
        self.time_stamp_half_hour[39267] = np.datetime64('2020-03-29T01:30:00.000000000')

        # add one more time step
        self.time_stamp_half_hour = np.append(self.time_stamp_half_hour, np.datetime64('2020-12-31T23:59:59.999999999'))
        self.time_stamp_daily = np.append(self.time_stamp_daily, np.datetime64('2020-12-31T23:59:59.999999999'))






def main():
    ...

if __name__ == '__main__':
    main()
