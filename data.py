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

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    df['Hour'] = df['timestamp'].apply(lambda row: row.hour)
    df['Day'] = df['timestamp'].apply(lambda row: row.day)
    df['Month'] = df['timestamp'].apply(lambda row: row.month)
    df['Year'] = df['timestamp'].apply(lambda row: row.year)
    df['is_weekend_holiday'] = df.apply(lambda row: is_holiday_weekend(row.is_holiday, row.is_weekend), axis=1)

    df['t_diff'] = df.apply(lambda row: row.t2 - row.t1, axis=1)


def data_analysis(df):
    season_names = ["fall", "spring", "summer", "winter"]
    print('describe output:')
    print(df.describe().to_string())
    print()
    print('corr output:')
    corr = df.corr()
    print(corr.to_string())
    print()
    corr_dict = corr.to_dict()
    corr_pairs = {}

    for i, (row_name, row_vals) in enumerate(zip(corr_dict.keys(), corr_dict.values())):
        for (column, corr_val) in list(zip(row_vals.keys(), row_vals.values()))[i + 1:]:
            pair = row_name, column
            corr_pairs[pair] = abs(corr_val)
            # print(f'{row_name} and {column} = {corr_val}')
    sorted_tuples = sorted(corr_pairs.items(), key=lambda item: item[1])
    corr_pairs_sorted = {k: v for k, v in sorted_tuples}  # lower to higher

    print("Highest correlated are: ")
    for index, (pair, corr) in enumerate(zip(reversed(sorted_tuples), pd.Series(corr_pairs_sorted).tail(5)[::-1])):
        print(f'{index + 1}. {pair[0]} with {corr:.6f}')
    print()
    print("Lowest correlated are: ")
    for index, (pair, corr) in enumerate(zip(sorted_tuples, pd.Series(corr_pairs_sorted).head(5))):
        print(f'{index + 1}. {pair[0]} with {corr:.6f}')
    print()

    df_grouped = df.groupby('season_name').mean()
    df_grouped_dict = df_grouped['t_diff'].to_dict()
    for season in season_names:
        print(f'{season} average t_diff is {df_grouped_dict[season]:.2f}')
    print(f'All average t_diff is {df.t_diff.mean():.2f}')
