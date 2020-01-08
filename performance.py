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
    
    fama_factors = datasets['Fama_French'][bets.clean_values_from_weights]
    
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
    
    #9. Stability of Wealth Process
    
    cum_log_returns = np.log1p(bets.strategy_daily_returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]
    res_dict['STABILITY_OF_WEALTH_PROCESS']=rhat**2
    

    
    ##############################################################################################
    # B. Performance measures
    #1. Equity curves
    def equity_curve(amount,ret):
        ret = hf.shift_array(ret,1,0)
        return amount*np.cumprod(1+ret)
    
    curves = dict()
    curves['Strategy']   = equity_curve(bets.starting_value,bets.strategy_daily_returns)
    curves['Buy & Hold Underlying'] = equity_curve(bets.starting_value,bets.underlying_daily_returns)
    curves['Benchmark']  = equity_curve(bets.starting_value,benchmark_returns)
    curves['Risk free Asset']  = equity_curve(bets.starting_value,rf_returns)
    curves['Long Contribution']       = equity_curve(bets.starting_value,bets.long_contribution)
    curves['Short Contribution']       = equity_curve(bets.starting_value,bets.short_contribution)
    
    
    #equity_curves = [strategy_curve,benchmark_curve,risk_free_curve]
#    equity_curves = [strategy_curve, benchmark_curve, underlying_curve]
    
    plots.equity_curves_plot(cleaned_index, curves)
    
    
    
    #2. Pnl from long positions check long_pnl 
    res_dict['PNL_FROM_STRATEGY'] = curves['Strategy'][-1]
    res_dict['PNL_FROM_LONG']    = curves['Long Contribution'][-1]
    
    #3. Annualized rate of return (Check this)
    res_dict['ANNUALIZED_AVERAGE_RATE_OF_RETURN'] = ((1+np.mean(bets.strategy_daily_returns))**(365)-1)
    res_dict['CUMMULATIVE_RETURN']= (np.cumprod(1+bets.strategy_daily_returns)[-1]-1)
    
    yrs = res_dict['TOTAL_BARS']/252
    res_dict['CAGR_STRATEGY'] = ((curves['Strategy'][-1]/curves['Strategy'][0])**(1/yrs))-1
    res_dict['CAGR_BENCHMARK'] = ((curves['Benchmark'][-1]/curves['Benchmark'][0])**(1/yrs))-1
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
        
    #4. Omega Ratio 
    returns_less_thresh = excess_returns-(((100)**(1/252))-1)
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    res_dict['OMEGA_RATIO']=numer/denom
    
    #5. Tail Ratio
    res_dict['TAIL_RATIO']=np.abs(np.percentile(bets.strategy_daily_returns, 95)) /np.abs(np.percentile(bets.strategy_daily_returns, 5))
    
    
    #6. Rachev Ratio 
    left_threshold = np.percentile(excess_returns, 5)
    right_threshold = np.percentile(excess_returns, 95)
    CVAR_left = -1*(np.nanmean(excess_returns[excess_returns<=left_threshold]))
    CVAR_right = (np.nanmean(excess_returns[excess_returns>=right_threshold]))
    res_dict['RACHEV_RATIO']=CVAR_right/CVAR_left
    #############################################################################################
    # E. RISK MEASURES
    
    #1. SKEWNESS, KURTOSIS
    res_dict['SKEWNESS'] = stats.skew(bets.strategy_daily_returns, bias = False)
    res_dict['KURTOSIS'] = stats.kurtosis(bets.strategy_daily_returns, bias = False)
    
    #2. ANNUALIZED VOLATILITY
    res_dict['ANNUALIZED_VOLATILITY'] = np.std(bets.strategy_daily_returns)*np.sqrt(252)

    #3. MAR Ratio
    res_dict['MAR_RATIO']=(res_dict['CAGR_STRATEGY'])/abs(res_dict['MDD_STRATEGY'])
    
    
    
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




    #############################################################################################
    # G. Factor Analysis
    import statsmodels.formula.api as sm # module for stats models
    from statsmodels.iolib.summary2 import summary_col
    
    def assetPriceReg(excess_ret, fama):
        
        df_stock_factor = pd.DataFrame({'ExsRet':excess_ret, 'MKT':fama[:,0], 'SMB':fama[:,1],'HML':fama[:,2], 'RMW':fama[:,3],'CMA':fama[:,4]})
        
        CAPM = sm.ols(formula = 'ExsRet ~ MKT', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
        FF3 = sm.ols( formula = 'ExsRet ~ MKT + SMB + HML', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
        FF5 = sm.ols( formula = 'ExsRet ~ MKT + SMB + HML + RMW + CMA', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})

        CAPMtstat = CAPM.tvalues
        FF3tstat = FF3.tvalues
        FF5tstat = FF5.tvalues
    
        CAPMcoeff = CAPM.params
        FF3coeff = FF3.params
        FF5coeff = FF5.params
    
        # DataFrame with coefficients and t-stats
        results_df = pd.DataFrame({'CAPMcoeff':CAPMcoeff,'CAPMtstat':CAPMtstat,
                                   'FF3coeff':FF3coeff, 'FF3tstat':FF3tstat,
                                   'FF5coeff':FF5coeff, 'FF5tstat':FF5tstat},
        index = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])
    
    
        dfoutput = summary_col([CAPM,FF3, FF5],stars=True,float_format='%0.4f',
                      model_names=['CAPM','Fama-French 3 Factors','Fama-French 5 factors'],
                      info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                 'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}, 
                                 regressor_order = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])
    
        print(dfoutput)
        
        return dfoutput,results_df
    
    res_dict['FACTOR_RES'],_= assetPriceReg(excess_returns, fama_factors)

   


    #############################################################################################
    # H. Bootstrap Stats
    # 1. Sharpe Bootstrap
    from arch.bootstrap import MovingBlockBootstrap
    from numpy.random import RandomState  
    bs_sharpe = MovingBlockBootstrap(5,excess_returns, random_state=RandomState(1234))
    
    def sharpe(y):
        return (mth.sqrt(252)*np.mean(y))/np.std(y)
    res = bs_sharpe.apply(sharpe,10000)      
    plots.density_plot_bootstrap(res,res_dict['SHARPE_RATIO'])
    


############################################################################################





    return res_dict
    
