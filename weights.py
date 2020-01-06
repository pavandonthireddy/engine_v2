# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 23:33:51 2019

@author: Pavan
"""
import data
import numpy as np
from settings import dollar_neutral, long_leverage, short_leverage, strategy_expression
import warnings
from expression import *
warnings.filterwarnings("ignore")


datasets = data.datasets_dict

for key, val in datasets.items():
    vars()[key] = val
    
    
strategy = eval(strategy_expression)


def weights(strategy,dollar_neutral=False,long_leverage=0.5,short_leverage=0.5):
    if dollar_neutral:
        exp_ls = strategy-np.nanmean(strategy,axis=1)[:,np.newaxis]
    else:
        exp_ls = strategy
    exp_norm = (exp_ls/np.nansum(abs(exp_ls),axis=1)[:,np.newaxis])
    exp_norm = np.where(exp_norm>=0,2*long_leverage*exp_norm,2*short_leverage*exp_norm)
    return exp_norm

strategy_weights = weights(strategy,dollar_neutral,long_leverage,short_leverage)

#strategy_weights = exp.sma(strategy_weights,3)