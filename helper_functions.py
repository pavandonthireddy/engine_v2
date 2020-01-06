# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:54:52 2019

@author: Pavan
"""

import pandas as pd
import numpy as np

def excel_list(file_name,column_name,sheet=0):
    List = pd.read_excel(file_name,sheet_name=sheet)
    List = List[column_name].tolist()
    return List

def shift(array, n,fill):
    shifted_array = np.empty_like(array)
    if n >= 0:
        shifted_array[:n,:] = fill
        shifted_array[n:,:] = array[:-n,:]
    else:
        shifted_array[n:,:] = fill
        shifted_array[:n,:] = array[-n:,:]
    return shifted_array

def shift_array(array,n,fill):
    shifted_array = np.empty_like(array)
    if n >= 0:
        shifted_array[:n] = fill
        shifted_array[n:] = array[:-n]
    else:
        shifted_array[n:] = fill
        shifted_array[:n] = array[-n:]
    return shifted_array
    

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)