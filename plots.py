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