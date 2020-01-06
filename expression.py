# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:45:19 2019

@author: Pavan
"""

import numpy as np
import scipy.stats as sc
import math


################################################################################################
# Helper Functions
def column_wise(fun,*args):
    return np.apply_along_axis(fun,0,*args)

def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

def fisher_helper_ema(array,alpha,window):
    def numpy_ewma(a, alpha, windowSize):
#        a=a.flatten()
        wghts = (1-alpha)**np.arange(windowSize)
#        wghts /= wghts.sum()
        out = np.convolve(a,wghts)
        out[:windowSize-1] = np.nan
        return out[:a.size] 
    return column_wise(numpy_ewma,array,alpha,window)

################################################################################################
# A. Mathematical Functions
# Rolling Median
def median(a,window):
    me = np.empty_like(a)
    me[:window-1,:]=np.nan
    me[window-1:,:]= np.squeeze(np.median(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return me

# Rolling Mean
def mean(a,window):
    me = np.empty_like(a)
    me[:window-1,:]=np.nan
    me[window-1:,:]= np.squeeze(np.mean(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return me
#Rolling Standard Deviation 
def stdev(a,window):
    std = np.empty_like(a)
    std[:window-1,:]=np.nan
    std[window-1:,:]= np.squeeze(np.std(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return std
#Rolling Product 
def product(a,window):
    prod = np.empty_like(a)
    prod[:window-1,:]=np.nan
    prod[window-1:,:]= np.squeeze(np.prod(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return prod

#Rolling Summation
def summation(a,window):
    summ = np.empty_like(a)
    summ[:window-1,:]=np.nan
    summ[window-1:,:]= np.squeeze(np.sum(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return summ

#Rolling Nan Product. Treats nan values as 1
def nanprod(a,window):
    nanprod = np.empty_like(a)
    nanprod[:window-1,:]=np.nan
    nanprod[window-1:,:]= np.squeeze(np.nanprod(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return nanprod

#Rolling Nan Summation. Treats nan values as 0 
def nansum(a,window):
    summ = np.empty_like(a)
    summ[:window-1,:]=np.nan
    summ[window-1:,:]= np.squeeze(np.nansum(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return summ

#Rolling Cummulative Product 
def cumproduct(a,window):
    prod = np.empty_like(a)
    prod[:window-1,:]=np.nan
    prod[window-1:,:]= np.squeeze(np.cumprod(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return prod

#Rolling Summation
def cumsummation(a,window):
    summ = np.empty_like(a)
    summ[:window-1,:]=np.nan
    summ[window-1:,:]= np.squeeze(np.cumsum(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return summ

#Rolling nan Cummulative Product. Treats nan as 1
def nancumproduct(a,window):
    prod = np.empty_like(a)
    prod[:window-1,:]=np.nan
    prod[window-1:,:]= np.squeeze(np.nancumprod(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return prod

#Rolling nan Cummulative Summation. Treats nan as 0 
def nancumsummation(a,window):
    summ = np.empty_like(a)
    summ[:window-1,:]=np.nan
    summ[window-1:,:]= np.squeeze(np.nancumsum(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return summ

#backward difference: a[n]=b[n]-b[n-(window-1)]
def back_diff(a,window):
    back= np.empty_like(a)
    back[:window-1,:]=np.nan
    back[window-1:,:]=a[window-1:,:]-a[:-(window-1),:]
    return back

# rolling integration 
def integrate(a,window):
    inte= np.empty_like(a)
    inte[:window-1,:]=np.nan
    inte[window-1:,:]=np.squeeze(np.trapz(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return inte

# rolling integration 
def integrate_along_x(y,x,window):
    inte= np.empty_like(y)
    inte[:window-1,:]=np.nan
    inte[window-1:,:]=np.squeeze(np.trapz(rolling_window(y,(window,y.shape[1])),rolling_window(x,(window,x.shape[1])),axis=2),axis=1)
    return inte

#Centers Value by subtracting its median over the TimePeriod. 
#Using the median instead of the mean reduces the effect of outliers.
def center(a,window):
    cen = np.empty_like(a)
    cen[:window-1,:]=np.nan
    cen[window-1:,:]= a[window-1:,:]-np.squeeze(np.median(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return cen

#Compresses Value to the -100...+100 range. For this, Value is divided by its interquartile range
# - the difference of the 75th and 25th percentile - taken over TimePeriod, and 
# then compressed by a cdf function. Works best when Value is an oscillator that crosses 
# the zero line. Formula: 200 * cdf(0.25*Value/(P75-P25)) - 100.
def compress(a,window):
    from scipy.stats import norm
    com = np.empty_like(a)
    com[:window-1,:]=np.nan
    value = a[window-1:,:]
    q25 = np.squeeze(np.quantile(rolling_window(a,(window,a.shape[1])),0.25,axis=2),axis=1)
    q75 = np.squeeze(np.quantile(rolling_window(a,(window,a.shape[1])),0.75,axis=2),axis=1)
    com[window-1:,:] = 200*(norm.cdf((0.25*value/(q75-q25))))-100
    return com

#Centers and compresses Value to the -100...+100 scale. 
#The deviation of Value from its median is divided by its interquartile range and then 
#compressed by a cdf function. Formula: 200 * cdf(0.5*(Value-Median)/(P75-P25)) - 100.
def scale(a,window):
    from scipy.stats import norm
    scale = np.empty_like(a)
    scale[:window-1,:]=np.nan
    value = a[window-1:,:]
    median = np.squeeze(np.median(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    q25 = np.squeeze(np.quantile(rolling_window(a,(window,a.shape[1])),0.25,axis=2),axis=1)
    q75 = np.squeeze(np.quantile(rolling_window(a,(window,a.shape[1])),0.75,axis=2),axis=1)
    scale[window-1:,:] = 200*(norm.cdf((0.25*(value-median)/(q75-q25))))-100
    return scale

#Normalizes Value to the -100...+100 range through subtracting its minimum and dividing 
#by its range over TimePeriod. Formula: 200 * (Value-Min)/(Max-Min) - 100 . 
#For a 0..100 oscillator, multiply the returned value with 0.5 and add 50.
def normalize(a,window):
    norm = np.empty_like(a)
    norm[:window-1,:]=np.nan
    value = a[window-1:,:]
    minimum = np.squeeze(rolling_window(a,(window,a.shape[1])).min(axis=2),axis=1)
    maximum = np.squeeze(rolling_window(a,(window,a.shape[1])).max(axis=2),axis=1)
    norm[window-1:,:] = 200*((value-minimum)/(maximum-minimum))-100
    return norm

def normalize_o(a,window):
    norm = np.empty_like(a)
    norm[:window-1,:]=np.nan
    value = a[window-1:,:]
    minimum = np.squeeze(rolling_window(a,(window,a.shape[1])).min(axis=2),axis=1)
    maximum = np.squeeze(rolling_window(a,(window,a.shape[1])).max(axis=2),axis=1)
    norm[window-1:,:] = 2*((value-minimum)/(maximum-minimum))-1
    return norm

#Calculates the Z-score of the Value. The Z-score is the deviation from
# the mean over the TimePeriod, divided by the standard deviation. 
# Formula: (Value-Mean)/StdDev.
def zscore(a,window):
    z = np.empty_like(a)
    z[:window-1,:]=np.nan
    value = a[window-1:,:]
    mean =np.squeeze(np.mean(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    stddev = np.squeeze(np.std(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    z[window-1:,:] = (value-mean)/stddev
    return z

#### Mathematical functions
def absolute(a):
    return np.absolute(a)

def sin(a):
    return np.sin(a)
def cos(a):
    return np.cos(a)
def tan(a):
    return np.tan(a)
def asin(a):
    return np.arcsin(a)
def acos(a):
    return np.arccos(a)
def atan(a):
    return np.arctan(a)
def sinh(a):
    return np.sinh(a)
def cosh(a):
    return np.cosh(a)
def tanh(a):
    return np.tanh(a)
def asinh(a):
    return np.arcsinh(a)
def acosh(a):
    return np.arccosh(a)
def atanh(a):
    return np.arctanh(a)

def floor(a):
    return np.floor(a)
def ceil(a):
    return np.ceil(a)
def clamp(a,minimum,maximum):
    return np.clip(a,minimum,maximum)
def around(a,decimals=0):
    return np.around(a,decimals)
def round_(a,decimals=0):
    return np.round_(a,decimals)
def rint(a):
    return np.rint(a)
def fix(a):
    return np.fix(a)
def trunc(a):
    return np.trunc(a)


def pdf(a):
    from scipy.stats import norm
    return norm.pdf(a)
def logpdf(a):
    from scipy.stats import norm
    return norm.logpdf(a)
def cdf(a):
    from scipy.stats import norm
    return norm.cdf(a)
def logcdf(a):
    from scipy.stats import norm
    return norm.logcdf(a)
def qnorm(a):
    from scipy.stats import norm
    return norm.ppf(a)
def survival(a):
    from scipy.stats import norm
    return norm.sf(a)
def inv_survival(a):
    from scipy.stats import norm
    return norm.isf(a)
def errorf(a):
    from scipy.special import erf
    return erf(a)
def errorfc(a):
    from scipy.special import erfc
    return erfc(a)



def exp(a):
    return np.exp(a)
def exp1(a):
    return np.exp1(a)
def exp2(a):
    return np.exp2(a)
def log(a):
    return np.log(a)
def log10(a):
    return np.log10(a)
def log2(a):
    return np.log2(a)
def log1p(a):
    return np.log1p(a)


def add(a,b):
    return np.add(a,b)
def receiprocal(a):
    return np.reciprocal(a)
def negative(a):
    return np.negative(a)
def multiply(a,b):
    return np.multiply(a,b)    
def divide(a,b):
    return np.divide(a,b)
def power(a,b):
    return np.power(a,b)
def subtract(a,b):
    return np.subtract(a,b)
def true_divide(a,b):
    return np.true_divide(a,b)
def remainder(a,b):
    return np.remainder(a,b)
def sqrt(a):
    return np.sqrt(a)
def square(a):
    return np.square(a)
def sign(a):
    return np.sign(a)
def maximum(a,b):
    return np.maximum(a,b)
def minimum(a,b):
    return np.minimum(a,b)
def nan_to_num(a):
    return np.nan_to_num(a)


############################################################################################
### Time Series properties, transformations and statistics
    
#Rolling Pearson Correlation   
def corr(a,b,window):
    from skimage.util import view_as_windows
    
    A = view_as_windows(a,(window,1))[...,0]
    B = view_as_windows(b,(window,1))[...,0]
    
    A_mA = A - A.mean(-1, keepdims=True)
    B_mB = B - B.mean(-1, keepdims=True)
    
#    ## Sum of squares across rows
#    ssA = (A_mA**2).sum(-1) # or better : np.einsum('ijk,ijk->ij',A_mA,A_mA)
#    ssB = (B_mB**2).sum(-1) # or better : np.einsum('ijk,ijk->ij',B_mB,B_mB)
    ssA = np.einsum('ijk,ijk->ij',A_mA,A_mA)
    ssB = np.einsum('ijk,ijk->ij',B_mB,B_mB)
    
    ## Finally get corr coeff
    out = np.full(a.shape, np.nan)
    out[window-1:] = np.einsum('ijk,ijk->ij',A_mA,B_mB)/np.sqrt(ssA*ssB)
    return out

# Rolling Covariance 
def covariance(a,b,window):
    from skimage.util import view_as_windows
    
    A = view_as_windows(a,(window,1))[...,0]
    B = view_as_windows(b,(window,1))[...,0]
    
    A_mA = A - A.mean(-1, keepdims=True)
    B_mB = B - B.mean(-1, keepdims=True)

    out = np.full(a.shape, np.nan)
    out[window-1:] = np.einsum('ijk,ijk->ij',A_mA,B_mB)
    return out

## Fisher transformation 
    
#Fisher Transform; transforms a normalized Data series to a normal distributed range.
# The return value has no theoretical limit, but most values are between -1 .. +1. 
# All Data values must be in the -1 .. +1 range f.i. by normalizing with
# the AGC, Normalize, or cdf function. The minimum Data length is 1.
 
def fisher(a):
    tran = np.clip(a,-0.998,0.998)
    return 0.5*np.log((1+tran)/(1-tran))    

def invfisher(a):
    return (np.exp(2*a)-1)/(np.exp(2*a)+1)

# Simple Moving average
def sma(array,window):
    def sma_array(array,window):
        weights = np.ones(window)/window
        ma = np.full_like(array,np.nan)
        ma[window-1:] = np.convolve(array, weights, 'valid')
        return ma
    return column_wise(sma_array,array,window)

def ema_v1(array,alpha,window):
    def numpy_ewma(a, alpha, windowSize):
        wghts = (1-alpha)**np.arange(windowSize)
        wghts /= wghts.sum()
        out = np.convolve(a,wghts)
        out[:windowSize-1] = np.nan
        return out[:a.size] 
    return column_wise(numpy_ewma,array,alpha,window)

def ema(array,window):
    def ExpMovingAverage(values, window):
        alpha = 2/(1.0+window)
        weights = (1-alpha)**np.arange(window)
        weights /= weights.sum()
        a =  np.convolve(values, weights, mode='full')[:len(values)]
        a[:window-1] = np.nan
        return a
    return column_wise(ExpMovingAverage,array,window)



### Check this AGAIN. this involves ema with 0.67 and 0.33 as weights
def fisher_norm(a,window):
    return fisher_helper_ema(fisher(ema(normalize_o(a,window),2)),2/3,2)


#Highest value over a specified period.
def maxval(a,window):
    z = np.full_like(a,np.nan)
    z[window-1:,:] = np.squeeze(rolling_window(a,(window,a.shape[1]))).max(axis=1)
    return z
#Lowest value over a specified period.
def minval(a,window):
    z = np.full_like(a,np.nan)
    z[window-1:,:] = np.squeeze(rolling_window(a,(window,a.shape[1]))).min(axis=1)
    return z

#Index of highest value over a specified period. 0 = highest value is at 
#current bar, 1 = at one bar ago, and so on. 
def max_index(a,window):
    z = np.full_like(a,np.nan)
    z[window-1:,:] = window-np.argmax(np.squeeze(rolling_window(a,(window,a.shape[1]))),axis=1)
    return z

#Index of lowest value over a specified period. 0 = lowest value is at 
#current bar, 1 = at one bar ago, and so on.
def min_index(a,window):
    z = np.full_like(a,np.nan)
    z[window-1:,:] = window-np.argmin(np.squeeze(rolling_window(a,(window,a.shape[1]))),axis=1)
    return z



    
#Fractal dimension of the Data series; normally 1..2. 
#Smaller values mean more 'jaggies'. Can be used to detect the current market regime or 
#to adapt moving averages to the fluctuations of a price series. 
#Requires a lookback period of twice the TimePeriod.   

def fractal_d(a,window):
    def fractal_d_helper_max(a,window):
        first = np.full_like(a,np.nan)
        second = np.full_like(a,np.nan)
        br = int(max(1,window/2))
        strides = np.squeeze(rolling_window(a,(window,a.shape[1])))
        first_half = strides[:,:br,:]
        second_half = strides[:,br:,:]
        first[window-1:,:]= first_half.max(axis=1)
        second[window-1:,:]= second_half.max(axis=1)
        return first,second
    
    def fractal_d_helper_min(a,window):
        first = np.full_like(a,np.nan)
        second = np.full_like(a,np.nan)
        br = int(max(1,window/2))
        strides = np.squeeze(rolling_window(a,(window,a.shape[1])))
        first_half = strides[:,:br,:]
        second_half = strides[:,br:,:]
        first[window-1:,:]= first_half.min(axis=1)
        second[window-1:,:]= second_half.min(axis=1)
        return first,second
    if window%2==0:
        period1 = int(max(1,window/2))
        period2 = int(max(1,window-period1))
        max_first,max_second = fractal_d_helper_max(a,window)
        min_first,min_second = fractal_d_helper_min(a,window)
        
        N1 = (max_first - min_first)/period1
        N2 = (max_second - min_second)/period2
        N3 = (maxval(a,window)-minval(a,window))/window
        nu = N1+N2
        nu[nu<=0]=1
        N3[N3<=0]=1
        return (log(N1+N2)-log(N3))/log(2)
    else:
        print('Time Period for fractional dimension should be an even number')


#4 pole Gauss Filter, returns a weighted average of the data within the given 
#time period, with the weight curve equal to the Gauss Normal Distribution. 
#Useful for removing noise by smoothing raw data.
# The minimum length of the Data series is equal to TimePeriod, 
# the lag is half the TimePeriod.
        
def gauss_filter(array,window):
    def gauss(a0,window):
#        a0=a0.flatten()
        N = len(a0)
        a0=a0[~np.isnan(a0)]
        to_fill = N-len(a0)
        poles = 4    
        PI = math.pi    
        beta = (1 - math.cos(2 * PI / window)) / (math.pow(2, 1 / poles) - 1)
        alpha = -beta + math.sqrt(math.pow(beta, 2) + 2 * beta)
        
        fil = np.zeros(4+len(a0))
        coeff = np.array([alpha**4,4*(1-alpha),-6*(1-alpha)**2,4*(1-alpha)**3,-(1-alpha)**4])
        for i in range(len(a0)):
            val = np.array([np.asscalar(a0[i]),fil[3+i],fil[2+i],fil[1+i],fil[i]])
            fil[4+i]=np.dot(coeff,val)
        if to_fill!=0:
            out = np.insert(fil[4:],0,np.repeat(np.nan,to_fill))
        else:
            out = fil[4:]
        return out
    return column_wise(gauss,array,window)
    
    
# Ehlers' smoothing filter, 2-pole Butterworth * SMA  
def smooth(array,cutoff=10):
    def smooth_helper(a0,cutoff=10):
        a0=a0.flatten()
        N = len(a0)
        a0=a0[~np.isnan(a0)]
        to_fill = N-len(a0)
       
        PI = math.pi   
        f = (math.sqrt(2)*PI)/cutoff
        a = math.exp(-f)
        c2 = 2*a*math.cos(f)
        c3 = -a*a
        c1 = 1-c2-c3
        
        src = np.insert(a0,0,0)
        coeff = np.array([0.5*c1,0.5*c1,c2,c3])
        fil = np.zeros(2+len(a0))
        for i in range(len(a0)):
            val = np.array([np.asscalar(src[i,]),np.asscalar(src[i+1,]),fil[i+1],fil[i]])
            fil[2+i]=np.dot(coeff,val)
        if to_fill!=0:
            out = np.insert(fil[2:],0,np.repeat(np.nan,to_fill))
        else:
            out = fil[2:]
        return out
    return column_wise(smooth_helper,array,cutoff)

def hurst_exp(array,window):
    window = max(window,20)
    hurst = (2.0-fractal_d(array,window))
    return clamp(smooth(hurst,int(window/10)),0,1)


#Linear Regression, also known as the "least squares moving average" (LSMA). 
#Linear Regression attempts to fit a straight trendline between several data points
# in such a way that the distance between each data point and the trendline
# is minimized. For each point, the straight line over the specified 
# previous bar period is determined in terms of y = b + m*x, where 
# y is the price and x the bar number starting at TimePeriod bars ago. 
# The formula for calculating b and m is then
#
#m = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²)
#b = (Σy - bΣx) / n
#
#where n is the number of data points (TimePeriod) and Σ is the summation operator.
# The LinearReg function returns b+m*(TimePeriod-1), i.e. the y of the current bar.  
def linear_reg(a,window):
    res = np.full_like(a,np.nan) 
    Y = np.squeeze(rolling_window(a,(window,a.shape[1])))
    X = np.tile(np.linspace(1,window,window)[:,np.newaxis],(Y.shape[0],1,a.shape[1]))     
    XY = np.multiply(X,Y)      
    s_XY = np.sum(XY,axis=1)
    s_Y = np.sum(Y,axis=1)
    s_X = np.sum(X,axis=1)
    s_X2 = np.sum(X**2,axis=1)
    m = (window*(s_XY)-(s_X*s_Y))/(window*s_X2-(s_X)**2)
    b = (s_Y-m*s_X)/window
    projected = m*(window+1)+b
    res[window-1:,:]=projected
    return res

#Forecast of bth bar after window
def linear_reg_forecast(a,window,bar):
    res = np.full_like(a,np.nan) 
    Y = np.squeeze(rolling_window(a,(window,a.shape[1])))
    X = np.tile(np.linspace(1,window,window)[:,np.newaxis],(Y.shape[0],1,a.shape[1]))     
    XY = np.multiply(X,Y)      
    s_XY = np.sum(XY,axis=1)
    s_Y = np.sum(Y,axis=1)
    s_X = np.sum(X,axis=1)
    s_X2 = np.sum(X**2,axis=1)
    m = (window*(s_XY)-(s_X*s_Y))/(window*s_X2-(s_X)**2)
    b = (s_Y-m*s_X)/window
    projected = m*(window+bar)+b
    res[window-1:,:]=projected
    return res

def linear_reg_angle(a,window):
    res = np.full_like(a,np.nan) 
    Y = np.squeeze(rolling_window(a,(window,a.shape[1])))
    X = np.tile(np.linspace(1,window,window)[:,np.newaxis],(Y.shape[0],1,a.shape[1]))     
    XY = np.multiply(X,Y)      
    s_XY = np.sum(XY,axis=1)
    s_Y = np.sum(Y,axis=1)
    s_X = np.sum(X,axis=1)
    s_X2 = np.sum(X**2,axis=1)
    m = (window*(s_XY)-(s_X*s_Y))/(window*s_X2-(s_X)**2)
#    b = (s_Y-m*s_X)/window
#    projected = m*(window+1)+b
    res[window-1:,:]=atan(m)
    return res

def linear_reg_slope(a,window):
    res = np.full_like(a,np.nan) 
    Y = np.squeeze(rolling_window(a,(window,a.shape[1])))
    X = np.tile(np.linspace(1,window,window)[:,np.newaxis],(Y.shape[0],1,a.shape[1]))     
    XY = np.multiply(X,Y)      
    s_XY = np.sum(XY,axis=1)
    s_Y = np.sum(Y,axis=1)
    s_X = np.sum(X,axis=1)
    s_X2 = np.sum(X**2,axis=1)
    m = (window*(s_XY)-(s_X*s_Y))/(window*s_X2-(s_X)**2)
    res[window-1:,:]=m
    return res

def linear_reg_intercept(a,window):
    res = np.full_like(a,np.nan) 
    Y = np.squeeze(rolling_window(a,(window,a.shape[1])))
    X = np.tile(np.linspace(1,window,window)[:,np.newaxis],(Y.shape[0],1,a.shape[1]))     
    XY = np.multiply(X,Y)      
    s_XY = np.sum(XY,axis=1)
    s_Y = np.sum(Y,axis=1)
    s_X = np.sum(X,axis=1)
    s_X2 = np.sum(X**2,axis=1)
    m = (window*(s_XY)-(s_X*s_Y))/(window*s_X2-(s_X)**2)
    b = (s_Y-m*s_X)/window
    res[window-1:,:]=b
    return res

#Rolling Skewness
def skew(a,window):
    sk = np.empty_like(a)
    sk[:window-1,:]=np.nan
    sk[window-1:,:]= np.squeeze(sc.skew(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return sk

def kurtosis(a,window):
    ku = np.empty_like(a)
    ku[:window-1,:]=np.nan
    ku[window-1:,:]= np.squeeze(sc.kurtosis(rolling_window(a,(window,a.shape[1])),axis=2),axis=1)
    return ku


def central_moment(a,moment,window):
    ku = np.empty_like(a)
    ku[:window-1,:]=np.nan
    ku[window-1:,:]= np.squeeze(sc.moment(rolling_window(a,(window,a.shape[1])),moment,axis=2),axis=1)
    return ku


#Number of data ranges, given by their Low and High values, that lie completely 
#inside the interval from Min to Max within the given Length. 
#Can be used to calculate the distribution of prices or candles. 
#Low and High can be set to the same value for counting all values 
#in the interval, or swapped for counting all candles that touch 
#the interval. Returns a value of 1 to TimePeriod

def nums_in_range(a,min_val,max_val,window):
    nums = np.full_like(a,np.nan)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    nums[window-1:,:]= np.count_nonzero((min_val<=chk)&(chk<=max_val),axis=1)
    return nums

#Length of the last streak of rising or falling values in the Data series, 
#back to the given TimePeriod. For a rising sequence its length is returned, 
#for a falling sequence the negative length 
#(f.i. -2 when Data[3] < Data[2] > Data[1] > Data[0]). Range: 1..TimePeriod.
    
#Very very slow. Need to optimize this. 

def streak_length(a,window):
    streak = np.full_like(a,np.nan)
    def num_rise_fall_array(a):
        window = a.shape[0]
        for i in range(window-1):
            if(a[window-i-1]>=a[window-i-2]):
                break
        for j in range(window-1):
            if(a[window-j-1]<=a[window-j-2]):
                break
        return j-i
    def num_rise_fall(array):
        return column_wise(num_rise_fall_array,array)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    streak[window-1:,:]=np.apply_along_axis(num_rise_fall,1,chk)
    streak = np.where(streak==0,np.nan,streak)
    return streak


#Number of upwards or downwards Data changes by more than the
# given Threshold within the TimePeriod,
def num_up(a,window):
    num = np.full_like(a,np.nan)
    def up(array):
        return ((array[1:]-array[:-1])>0).sum()
    def up_2d(array):
        return column_wise(up,array)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    num[window-1:,:]=np.apply_along_axis(up_2d,1,chk)
    return num

def num_down(a,window):
    num = np.full_like(a,np.nan)
    def down(array):
        return ((array[1:]-array[:-1])<0).sum()
    def down_2d(array):
        return column_wise(down,array)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    num[window-1:,:]=np.apply_along_axis(down_2d,1,chk)
    return num


def sum_up(a,window):
    num = np.full_like(a,np.nan)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    res = chk[:,1:,:]-chk[:,:-1,:]
    res[res<0]=0
    num[window-1:,:]=np.nansum(res,axis=1)
    return num

def sum_down(a,window):
    num = np.full_like(a,np.nan)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    res = chk[:,1:,:]-chk[:,:-1,:]
    res[res>0]=0
    num[window-1:,:]=np.sum(res,axis=1)
    return num

def Profit_factor(a,window):
    up = sum_up(a,window)
    down = np.abs(sum_down(a,window))
    down = np.where((down==0)&(up==0),1,down)
    down = np.where((down==0)&(up!=0),10,down)
    p=np.divide(up,down)
    return clamp(p,0,9.999)


#Returns the given quantile of the Data series with given Length;
# f.i. q = 95 returns the Data value that is above 95% of all other values
# . q = 50 returns the Median of the Data series.
def quantile(a,window,q):
    scale = np.empty_like(a)
    scale[:window-1,:]=np.nan
    q_value = np.squeeze(np.quantile(rolling_window(a,(window,a.shape[1])),q,axis=2),axis=1)
    scale[window-1:,:] =q_value
    return scale

def rank(a):
    import scipy.stats.mstats as mstats
    keep = ~np.isnan(a).all(axis=1)
    if np.sum(keep)==0:
        return np.full_like(a,np.nan)
    else:
        array = np.ma.masked_invalid(a[keep])
        ranks = np.apply_along_axis(mstats.rankdata,1,array)
        ranks[ranks == 0] = np.nan
        ranks-=1
        ranks=ranks/array.shape[1]
        ranks = np.vstack((a[~keep],ranks))
        return ranks
#Time series rank. 
def ts_rank(a,window):
    ans = np.full_like(a,np.nan)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    def rankdata(c):
        c_clean = c[~np.isnan(c)]
        if len(c_clean)!=0:
            return sc.percentileofscore(c_clean,c_clean[-1])
        else:
            return np.nan
    def rankdata_2d(c):
        return column_wise(rankdata,c)
    ans[window-1:,:] = (np.apply_along_axis(rankdata,1,chk))
    return ans


def R2(a,window):
    ans = np.full_like(a,np.nan) 
    Y = np.squeeze(rolling_window(a,(window,a.shape[1])))
    X = np.tile(np.linspace(1,window,window)[:,np.newaxis],(Y.shape[0],1,a.shape[1]))     
    XY = np.multiply(X,Y)      
    s_XY = np.sum(XY,axis=1)
    s_Y = np.sum(Y,axis=1)
    s_X = np.sum(X,axis=1)
    s_X2 = np.sum(X**2,axis=1)
    m = (window*(s_XY)-(s_X*s_Y))/(window*s_X2-(s_X)**2)
    b = (s_Y-m*s_X)/window
    Y_bar = np.mean(Y,axis=1)[:,np.newaxis,:]
    tot = (Y-Y_bar)**2
    SS_tot = np.sum(tot,axis=1)
    M = m[:,np.newaxis,:]
    B = b[:,np.newaxis,:]
    res = (Y-(M*(X)+B))**2
    SS_res = np.sum(res,axis=1)
    ans[window-1:,:]=1-(SS_res/SS_tot)
    return ans

#Expected logarithmic gain rate of the Data series in the range of about +/-0.0005. 
#The gain rate is derived from the Shannon probability 
#P = (1 + Mean(Gain) / RootMeanSquare(Gain)) / 2, which is the likeliness of a rise 
#or fall of a high entropy data series in the next bar period. 
#A positive gain rate indicates that the series is more likely to rise, 
#a negative gain rate indicates that it is more likely to fall. 
#The zero crossover could be used for a trade signal
def shannon_gain(a,window):
    ans = np.full_like(a,np.nan)
    chk = np.squeeze(rolling_window(a,(window,a.shape[1])))
    G = np.log10(chk[:,:-1,:]/chk[:,1:,:])
    G[~np.isfinite(G)] = 0.0000001
    avgx = np.mean(G,axis=1)
    rmsx = np.sqrt(np.mean(np.square(G),axis=1))
    P = ((avgx/rmsx)+1)/2.0
    ans[window-1:,:] = np.log10(((1+rmsx)**P)*(1-rmsx)**(1-P))
    ans[ans==4.34294e-08]=np.nan
    return ans


###############################################################################################







#strategy_expression = -(gauss_filter(High,5)-gauss_filter(Close,5))
    
