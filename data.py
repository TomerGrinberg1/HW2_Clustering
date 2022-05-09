import pandas as pd
from datetime import datetime


def load_data(path):
    df = pd.read_csv(path)
    return df


def is_holiday_weekend(holiday, weekend):
    if holiday == 0:
        if weekend == 0:
            return 0
        else:
            return 1
    else:
        if weekend == 0:
            return 2
        else:
            return 3


def add_new_columns(df):
    season_names = ['spring', 'summer', 'fall', 'winter']
    df['season_name'] = df.apply(lambda row: season_names[row.season], axis=1)

    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['Hour'] = df['datetime'].apply(lambda row: row.hour)
    df['Day'] = df['datetime'].apply(lambda row: row.day)
    df['Month'] = df['datetime'].apply(lambda row: row.month)
    df['Year'] = df['datetime'].apply(lambda row: row.year)

    df['is_weekend_holiday'] = df.apply(lambda row: is_holiday_weekend(row.is_holiday, row.is_weekend), axis=1)

    df['t_diff'] = df.apply(lambda row: row.t2 - row.t1, axis=1)


def data_analysis(df):
    season_names = ['fall', 'spring', 'summer', 'winter']
    print('describe output:')
    print(df.describe().to_string())
    print()
    print('corr output:')
    corr = df.corr().abs()
    print(corr.to_string())
    print()

    corr_sorted = corr[corr < 1].unstack().transpose().sort_values(ascending=False).drop_duplicates()

    print("Highest correlated are: ")
    for count, (pair, corr) in enumerate(zip(corr_sorted.head(5).keys(), corr_sorted.head(5).to_dict().values())):
        print(f'{count + 1}. {pair} with {corr:.6f}')

    print("Lowest correlated are: ")
    for count, (pair, corr) in enumerate(zip(reversed(corr_sorted.tail(5).keys()),
                                             reversed(corr_sorted.tail(5).to_dict().values()))):
        print(f'{count + 1}. {pair} with {corr:.6f}')

    df_grouped = df.groupby('season_name').mean()
    df_grouped_dict = df_grouped['t_diff'].to_dict()
    for season in season_names:
        print(f'{season} average t_diff is {df_grouped_dict[season]:.2f}')
    print(f'All average t_diff is {df.t_diff.mean():.2f}')