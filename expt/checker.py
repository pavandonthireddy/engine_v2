# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:55:43 2019

@author: Pavan
"""

import numpy as np
import math
np.random.seed(5)
import scipy.stats as sc
#
#
#datasets = data.datasets_dict
#
#High = datasets['High']
#Close = datasets['Close']
#Low = datasets['Low']
#Open = datasets['Open']

a = np.random.rand(200,5)
#a[:2,:]=np.nan
window=10

def column_wise(fun,*args):
    return np.apply_along_axis(fun,0,*args)

def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)



ans = np.apply_along_axis(sc.rankdata,1,a)/a.shape[1]



from nltk import CFG
from nltk.parse.generate import generate

#Define your grammar from string
#You can define it using other methods, but I only know this xD

grammar = CFG.fromstring("""
<expr>
<expr> -> (<var>) | (<fopbi> (<var>, <day>)) |(<fopbi> (<expr>, <day>))| (<fopun> (<expr>))| (<expr><matbi><expr>) 
<fopbi> -> mean | 
            median |   
            stdev| 
            back_diff| 
            center| 
            compress| 
            scale| 
            normalize_o|       
            zscore|
            corr| 
            covariance| 
            fisher| 
            invfisher|       
            sma|
            ema| 
            fisher_norm| 
            max_val| 
            min_val|       
            gauss_filter|
            smooth|
<fopun> -> rank| -1* | 1/
<matbi> -> + | - | * | /
<var> -> Open|High|Low|Close|Volume
<day> ::=  5 | 6 | 10 | 15 | 20
""")

#grammar = CFG.fromstring("""
#  S -> NP VP
#  VP -> V NP
#  V -> "mata" | "roba"
#  NP -> Det N | NP NP
#  Det -> "un" | "el" | "con" | "a" | "una"
#  N -> "bebé" | "ladrón" | "Obama" | "perrete" | "navastola" | "navaja" | "pistola" """)

''' This grammar creates sentences like:
        El bebé roba a Obama
        Baby steals Obama (in spanish)
'''
#With this we "create" all the possible combinations
grammar.productions()

#Here you can see all the productions (sentences) with 5 words
#created with this grammar
for production in generate(grammar, depth=6):
    print(' '.join(production))

    






