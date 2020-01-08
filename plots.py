# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:21:21 2019

@author: Pavan
"""

import datetime as dt
import helper_functions as hf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



    
def equity_curves_plot(x,y_dict):  
    from matplotlib.ticker import FuncFormatter
    import matplotlib.pyplot as plt
    formatter = FuncFormatter(hf.millions)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)
    for name,curve in y_dict.items():
        if name == 'Strategy':
            plt.plot(x,curve,'black',label = name)
        if name == 'Benchmark':
            plt.plot(x,curve,'lightgray',label = name)
    plt.legend(loc="upper left")
    plt.title("Equity Curve") 
    plt.xlabel("Year") 
    plt.ylabel("Dollars") 
    plt.grid('on')
    plt.show()
#    plt.cla()
    

def density_plot_bootstrap(bootstrap,res):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots() 
    sns.distplot(bootstrap, hist=False, kde=True, bins=int(180/5),kde_kws={'shade': True,'linewidth': 4})
    plt.axvline(x=res, color='black',label ='Strategy_Sharpe = {}'.format(res))
    plt.axvline(x=np.median(res), color='k', linestyle='--', label ='Bootstrap Sharpe Median = {}'.format(np.median(bootstrap)))
    plt.legend(loc="upper left")
    plt.xlabel('Sharpe')
    plt.title('Distribution of Sharpe Ratio')
    plt.grid('on')
    plt.show()