import pandas as pd
import vectorbt as vbt  
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np



def get_data(ticker, startDate, endDate, interval):
    data = vbt.YFData.download(symbols=ticker, start=startDate, end=endDate, missing_index='drop',
                                 interval=interval).get('Close')
    
    return data
    
def clean_data(data): 
    data.index = pd.to_datetime(data.index)
    # data.dropna(axis=1)
    data.ffill(axis=0,inplace=True)
    data.dropna(axis=1, inplace=True)

    return data


def plot_linreg(benchmark, asset, data):
    sns.pairplot(data, y_vars=asset, x_vars=benchmark, 
                    height=2, aspect=2 , kind='reg')
    

def beta_value(data, benchmark):
    df = pd.DataFrame({'a':[1]})
    for i in data.keys():    
        slope, intercept, r, p, std = stats.linregress(data[benchmark].dropna(), data[i].dropna())
        df.insert(1, i, slope)
    df.drop(labels='a', axis=1,inplace=True)
    df.index = ['beta']

    return df
    

def main():
    tickers = ["^BVSP", "VALE", "PETR4.SA", "WEGE3.SA", "BBAS3.SA"]
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=700)
    interval = '1D'
    benchmark = "^BVSP"  ## data for independent variable in linear regression 

    ### CREATE CSV FILE ### 
    # df = get_data(tickers, startDate=start_date, endDate=end_date, interval=interval)
    # df.to_csv('ativos_br.csv')
    df = pd.read_csv('ativos_br.csv', index_col='Date')

    df_clean = clean_data(df)
    df_clean = df_clean.pct_change()

    assets_name = [w for w in df_clean.keys() if w != benchmark]   ## LIST WITH ALL COLUMNS NAME WITHOUT BENCHMARK
    plot_linreg(benchmark, assets_name, df_clean)
    betas = beta_value(df_clean, benchmark) 
    print(betas.iloc[:,:])


main()