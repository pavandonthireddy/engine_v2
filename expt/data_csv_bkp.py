# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:47:58 2019

@author: Pavan
"""

import numpy as np
import pandas as pd 
import helper_functions as hf
import settings as st

#from settings import start_date, end_date, datasets_address, tickers_file_name, tickers_column_name, variables_file_name, variables_column_name, delay, portfolio_all, other_assets

start_date = st.start_date
end_date = st.end_date
datasets_address = st.datasets_address
tickers_file_name = st.tickers_file_name
tickers_column_name = st.tickers_column_name
variables_file_name = st.variables_file_name
variables_column_name = st.variables_column_name
delay = st.delay

portfolio_all = st.portfolio_all
other_assets = st.other_assets


variable_list    = hf.excel_list(variables_file_name,variables_column_name)


address=datasets_address

def get_clean_index(address,start_date,end_date):
    sample_address = address+"sample.csv"
    sample = pd.read_csv(sample_address,index_col=0)
    sample.index = pd.to_datetime(sample.index)
    sample = sample[sample.index.isin(pd.date_range(start=start_date,end=end_date))]
    sample = sample.dropna(axis=0,how='all')
    
    diff_dates = pd.DataFrame(index=sample.index)
    diff_dates['index']= diff_dates.index.astype('datetime64[ns]') 
    diff_dates['Diff'] = (diff_dates['index'] - diff_dates['index'].shift())
    tickers_universe = list(sample.columns)
    return sample.index, diff_dates['Diff'].values, tickers_universe

clean_index,diff_dates,tickers_universe = get_clean_index(datasets_address,start_date,end_date)
tickers = tickers_universe[50:75]

def get_portfolio(portfolio_all,tickers_universe,tickers):
    if portfolio_all:
        return tickers_universe
    else:
        return tickers

portfolio = get_portfolio(portfolio_all,tickers_universe,tickers)
    

def load_datasets(address,other_assets,variable_list,clean_index,portfolio,tickers_universe):
    datasets_dict = {}
    for var in variable_list:
        var_address=address+var+".csv"
        var_df=pd.read_csv(var_address,index_col=0)
        if var not in other_assets:
            var_df.columns = tickers_universe #white spaces issues; hence this
            var_df = var_df[portfolio]
        var_df.index = pd.to_datetime(var_df.index)
        var_df = var_df[var_df.index.isin(clean_index)]
        var_df.fillna(method='ffill',axis=0,inplace=True)
        var_array=var_df.values
        if var in other_assets and var !='Fama_French':
            var_array = var_array[:,6]
        datasets_dict[var]=var_array
    return datasets_dict
        
datasets_dict = load_datasets(datasets_address,other_assets,variable_list,clean_index,portfolio,tickers_universe)

Open_for_ret = hf.shift(datasets_dict['Open'],-1*delay,np.nan)
Close_for_ret = hf.shift(datasets_dict['Close'],-1*delay,np.nan) 
Volume_for_ret = hf.shift(datasets_dict['Volume'],-1*delay,np.nan)
High_for_ret = hf.shift(datasets_dict['High'],-1*delay,np.nan)
Low_for_ret = hf.shift(datasets_dict['Low'],-1*delay,np.nan)

