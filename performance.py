# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:49:17 2019

@author: Pavan
"""

def metrics():
    import bets
    import data
    import numpy as np
    import plots 
    import helper_functions as hf
    import pandas as pd 
    import math as mth
    from scipy import stats
    
    
    datasets = data.datasets_dict
    
    benchmark_returns = datasets['benchmark'][bets.clean_values_from_weights]
    rf_returns   = datasets['rf_rate'][bets.clean_values_from_weights]
    
    cleaned_index = bets.cleaned_index_weights
    
    
    res_dict = dict()
    ##############################################################################################
    #A. General Characterstics
    #1. Time range
    res_dict['START_DATE'] = cleaned_index.min()
    res_dict['END_DATE'] = cleaned_index.max()
    res_dict['TIME_RANGE_DAYS'] = ((cleaned_index.max()-cleaned_index.min()).astype('timedelta64[D]'))/np.timedelta64(1, 'D')
    #years = ((end_date-start_date).astype('timedelta64[Y]'))/np.timedelta64(1, 'Y')
    res_dict['TOTAL_BARS'] = len(cleaned_index)
    
    #2. Average AUM
    res_dict['AVERAGE_AUM'] = np.nanmean(np.nansum(np.abs(bets.dollars_at_open),axis=1))
    
    
    #3. Capacity of Strategy
    
    
    
    #4. Leverage (!!! Double check -something to do with sum of long_lev and short_lev > 1)
    res_dict['AVERAGE_POSITION_SIZE'] = np.nanmean(np.nansum(bets.dollars_at_open,axis=1))
    
    res_dict['NET_LEVERAGE'] = round(res_dict['AVERAGE_POSITION_SIZE']/res_dict['AVERAGE_AUM'],2)
    
    
    #5. Turnover
    daily_shares = np.nansum(bets.purchased_shares,axis=1)
    daily_value_traded = np.nansum(np.abs(bets.dollars_at_open),axis=1)
    daily_turnover = daily_shares/(2*daily_value_traded)
    res_dict['AVERAGE_DAILY_TURNOVER']= np.mean(daily_turnover)
    
    #6. Correlation to underlying
    res_dict['CORRELATION_WITH_UNDERLYING'] = np.corrcoef(bets.underlying_daily_returns,bets.strategy_daily_returns)[0,1]
    
    #7. Ratio of longs
    
    res_dict['LONG_RATIO'] = ((bets.cleaned_strategy_weights>0).sum())/(np.ones(bets.cleaned_strategy_weights.shape,dtype=bool).sum())
    
    #8. Maximum dollar position size 
    
    res_dict['MAX_SIZE'] = np.nanmax(np.abs(bets.cleaned_strategy_weights))
    
    ##############################################################################################
    # B. Performance measures
    #1. Equity curves
    def equity_curve(amount,ret):
        ret = hf.shift_array(ret,1,0)
        return amount*np.cumprod(1+ret)
    
    curves = dict()
    curves['STRATEGY_CURVE']   = equity_curve(bets.starting_value,bets.strategy_daily_returns)
    curves['UNDERLYING_CURVE'] = equity_curve(bets.starting_value,bets.underlying_daily_returns)
    curves['BENCHMARK_CURVE']  = equity_curve(bets.starting_value,benchmark_returns)
    curves['RISK_FREE_CURVE']  = equity_curve(bets.starting_value,rf_returns)
    curves['LONG_CURVE']       = equity_curve(bets.starting_value,bets.long_contribution)
    curves['SHORT_CURVE']       = equity_curve(bets.starting_value,bets.short_contribution)
    
    
    #equity_curves = [strategy_curve,benchmark_curve,risk_free_curve]
#    equity_curves = [strategy_curve, benchmark_curve, underlying_curve]
    
    plots.equity_curves_plot(cleaned_index, curves)
    
    
    
    #2. Pnl from long positions check long_pnl 
    res_dict['PNL_FROM_STRATEGY'] = curves['STRATEGY_CURVE'][-1]
    res_dict['PNL_FROM_LONG']    = curves['LONG_CURVE'][-1]
    
    #3. Annualized rate of return (Check this)
    res_dict['ANNUALIZED_AVERAGE_RATE_OF_RETURN'] = ((1+np.mean(bets.strategy_daily_returns))**(365)-1)
    res_dict['CUMMULATIVE_RETURN']= (np.cumprod(1+bets.strategy_daily_returns)[-1]-1)
    #4. Hit Ratio
    
    res_dict['HIT_RATIO'] = ((bets.daily_pnl>0).sum())/((bets.daily_pnl>0).sum()+(bets.daily_pnl<0).sum()+(bets.daily_pnl==0).sum())
    
    ##############################################################################################
    # C. Runs
    # 1. Runs concentration
    def runs(returns):
        wght=returns/returns.sum()
        hhi=(wght**2).sum()
        hhi=(hhi-returns.shape[0]**-1)/(1.-returns.shape[0]**-1)
        return hhi
    
    res_dict['HHI_PLUS'] = runs(bets.strategy_daily_returns[bets.strategy_daily_returns>0])
    res_dict['HHI_MINUS'] = runs(bets.strategy_daily_returns[bets.strategy_daily_returns<0])
    
    # 2. Drawdown and Time under Water

    
    
    def MDD(returns):
        def returns_to_dollars(amount,ret):
            return amount*np.cumprod(1+ret)
        
        doll_series = pd.Series(returns_to_dollars(100,returns))
        
        Roll_Max = doll_series.cummax()
        Daily_Drawdown = doll_series/Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        return Max_Daily_Drawdown
    
   
    DD_strategy=MDD(bets.strategy_daily_returns)
    DD_benchmark=MDD(benchmark_returns)
    res_dict['MDD_STRATEGY'] = DD_strategy.min()
    res_dict['MDD_BENCHMARK'] = DD_benchmark.min()

    
    
    #3. 95 percentile
    res_dict['95PERCENTILE_DRAWDOWN_STRATEGY']=DD_strategy.quantile(0.05)
    res_dict['95PERCENTILE_DRAWDOWN_BENCHMARK']=DD_benchmark.quantile(0.05)

    
    #############################################################################################
    # D. Efficiency
    
    #1. Sharpe Ratio
    excess_returns = bets.strategy_daily_returns-rf_returns
    res_dict['SHARPE_RATIO'] = mth.sqrt(252)*np.mean(excess_returns)/np.std(excess_returns)
    
    #from statsmodels.graphics.tsaplots import plot_acf
    #plot_acf(excess_returns)
    #2. sortino Ratio
    res_dict['SORTINO_RATIO'] = mth.sqrt(252)*np.mean(excess_returns)/np.std(excess_returns[excess_returns<np.mean(excess_returns)])

    
    #2.Probabilistic Sharpe ratio
    from scipy.stats import norm
    from scipy.stats import kurtosis, skew
    g_3 = skew(excess_returns)
    g_4 = kurtosis(excess_returns)
    res_dict['PROBABILISTIC_SHARPE_RATIO'] = norm.cdf(((res_dict['SHARPE_RATIO']-2)*mth.sqrt(len(excess_returns)-1))/(mth.sqrt(1-(g_3*res_dict['SHARPE_RATIO'])+(0.25*(g_4-1)*res_dict['SHARPE_RATIO']*res_dict['SHARPE_RATIO']))))
    
    #3.Information ratio
    excess_returns_benchmark = bets.strategy_daily_returns-benchmark_returns
    res_dict['INFORMATION_RATIO'] = mth.sqrt(252)*np.mean(excess_returns_benchmark)/np.std(excess_returns_benchmark)
    
    #3. t_stat & P-value
    m = np.mean(excess_returns)
    s = np.std(excess_returns)
    n = len(excess_returns)
    t_stat = (m/s)*mth.sqrt(n)
    res_dict['t_STATISTIC']= t_stat
    
    pval = stats.t.sf(np.abs(t_stat), n**2-1)*2 # Must be two-sided as we're looking at <> 0
    
    res_dict['p-VALUE']=pval
    if pval <= 0.0001:
        res_dict['SIGNIFICANCE_AT_0.01%']='STATISTICALLY_SIGNIFICANT'
    else:
        res_dict['SIGNIFICANCE_AT_0.01%']='NOT_STATISTICALLY_SIGNIFICANT'
    #############################################################################################
    # E. RISK MEASURES
    
    #1. SKEWNESS, KURTOSIS
    res_dict['SKEWNESS'] = stats.skew(bets.strategy_daily_returns, bias = False)
    res_dict['KURTOSIS'] = stats.kurtosis(bets.strategy_daily_returns, bias = False)
    
    #2. ANNUALIZED VOLATILITY
    res_dict['ANNUALIZED_VOLATILITY'] = np.std(bets.strategy_daily_returns)*np.sqrt(252)

    
    
    
    
    
    #############################################################################################
    # F. Classification scores
    
    sign_positions = np.sign(bets.purchased_shares).flatten()
    sign_profits = np.sign(bets.pnl).flatten()
    
    invalid = np.argwhere(np.isnan(sign_positions+sign_profits))
    
    sign_positions_final = np.delete(sign_positions, invalid)
    sign_profits_final = np.delete(sign_profits,invalid)
    
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(sign_profits_final, sign_positions_final)

    
    res_dict['CLASSIFICATION_DATA']= {'Class' :['-1','0','1'], 'Precision':list(precision),'Recall':list(recall),'F-Score':list(fscore),'Support':list(support) }
    res_dict['CLASSIFICATION_DATA']=pd.DataFrame(res_dict['CLASSIFICATION_DATA'])


    return res_dict

############################################################################################
# Bootstrapping

#from arch.bootstrap import MovingBlockBootstrap
#import matplotlib.pyplot as plt
#
#bs_sharpe = MovingBlockBootstrap(1,excess_returns)
#
#final_sharpe = np.empty(excess_returns.shape)
#for data in bs_sharpe.bootstrap(1000):
#    final_sharpe=np.vstack((final_sharpe,data[0][0]))
#final_sharpe= final_sharpe[1:,10:]
#
#mean_bs = np.mean(final_sharpe,axis=0)
#std_bs = np.std(final_sharpe,axis=0)
#sharpe_dist = mth.sqrt(252)*mean_bs/std_bs
#plt.hist(sharpe_dist)
    
