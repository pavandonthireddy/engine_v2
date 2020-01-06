# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:12:45 2019

@author: Pavan
"""

import numpy as np
from settings import starting_value, costs_threshold
from weights import strategy_weights, long_leverage,short_leverage
from data import delay,Open_for_ret, Close_for_ret, Volume_for_ret, clean_index, High_for_ret, Low_for_ret




def get_valid_index(strategy_weights,delay):
    valid_index = ~np.isnan(strategy_weights).all(axis=1)
    valid_index[-1*delay]=False
    return valid_index

clean_values_from_weights = get_valid_index(strategy_weights,delay)
cleaned_index_weights = (clean_index.values)[clean_values_from_weights]

cleaned_strategy_weights       = strategy_weights[clean_values_from_weights]


def bets_to_pnl(starting_value,strategy_weights,clean_values,o,h,l,c,long_lev, short_lev):
    

    cleaned_weights       = strategy_weights[clean_values]

    O  = (o)[clean_values]
    C = (c)[clean_values]
#   V   = (V)[clean_values]
    H  = (h)[clean_values]
    L   = (l)[clean_values]
    
    
    dollars_at_open  = np.zeros(cleaned_weights.shape)
    dollars_at_close = np.zeros(cleaned_weights.shape)
    purchased_shares = np.zeros(cleaned_weights.shape)
    costs            = np.zeros(cleaned_weights.shape)
    value_at_open    = np.zeros(cleaned_weights.shape[0])
    value_at_open[0] = starting_value
    value_at_close   = np.zeros(cleaned_weights.shape[0])
    pnl              = np.zeros(cleaned_weights.shape)
    daily_pnl        = np.zeros(cleaned_weights.shape[0])
    long_pnl        = np.zeros(cleaned_weights.shape[0])
    short_pnl        = np.zeros(cleaned_weights.shape[0])
    

    for i in range(dollars_at_open.shape[0]):
        dollars_at_open[i,:] = value_at_open[i]*cleaned_weights[i,:]
        purchased_shares[i,:]= dollars_at_open[i,:]/O[i,:]
        dollars_at_close[i,:]= C[i,:]*purchased_shares[i,:]
        costs[i,:]           = np.abs(purchased_shares[i,:])*(H[i,:]-L[i,:])*costs_threshold
        pnl[i,:]             = (dollars_at_close[i,:]-dollars_at_open[i,:]-costs[i,:])
        daily_pnl[i]         = np.nansum(pnl[i,:])
        long_pnl[i]         = np.nansum(pnl[i,:][cleaned_weights[i,:]>0])
        short_pnl[i]        = np.nansum(pnl[i,:][cleaned_weights[i,:]<0])
        value_at_close[i]    = np.nansum(np.abs(dollars_at_open[i,:])/(long_lev+short_lev))+np.nansum(pnl[i])
        if i != dollars_at_open.shape[0]-1:
            value_at_open[i+1]=value_at_close[i]
    
    strategy_daily_returns = daily_pnl/value_at_open
    long_contribution = long_pnl/value_at_open
    short_contribution = short_pnl/value_at_open

    return strategy_daily_returns, long_contribution, short_contribution, costs, purchased_shares,dollars_at_open, dollars_at_close, value_at_open, value_at_close, pnl, daily_pnl
    
strategy_daily_returns, long_contribution, short_contribution, costs, purchased_shares, dollars_at_open, dollars_at_close, value_at_open, value_at_close, pnl, daily_pnl = bets_to_pnl(starting_value,strategy_weights,clean_values_from_weights,Open_for_ret,High_for_ret,Low_for_ret,Close_for_ret, long_leverage, short_leverage)



underlying_daily_returns = np.nanmean(np.log(Close_for_ret[clean_values_from_weights]/Open_for_ret[clean_values_from_weights]),axis=1)






    
    
    
    
    

