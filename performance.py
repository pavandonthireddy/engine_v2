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
    from scipy.stats import norm
    from scipy.stats import kurtosis, skew
    
    
    datasets            = data.datasets_dict
    
    benchmark_returns   = datasets['benchmark'][bets.clean_values_from_weights]
    rf_returns          = datasets['rf_rate'][bets.clean_values_from_weights]    
    fama_factors        = datasets['Fama_French'][bets.clean_values_from_weights]
    
    cleaned_index       = bets.cleaned_index_weights
    excess_returns      = bets.strategy_log_returns-rf_returns
    
    
    res_dict = dict()
    
    ##############################################################################################
    #A. General Characterstics
    #1. Time 
    res_dict['START_DATE']      = hf.to_datetime(cleaned_index.min()).strftime("%d %B, %Y")
    
    res_dict['END_DATE']        = hf.to_datetime(cleaned_index.max()).strftime("%d %B, %Y")
    
    res_dict['TIME_RANGE_DAYS'] = '{0:.0f} days'.format(int(((cleaned_index.max()-cleaned_index.min()).astype('timedelta64[D]'))/np.timedelta64(1, 'D')))

    res_dict['TOTAL_BARS']      = '{0:.0f} bars'.format(int(len(cleaned_index)))
    

    sr = mth.sqrt(252)*np.mean(excess_returns)/np.std(excess_returns)
 

    def minTRL(sharpe, skew, kurtosis, target_sharpe=0, prob=0.95):
        from scipy.stats import norm
        min_track = (
            1
            + (1 - skew * sharpe + sharpe ** 2 * (kurtosis - 1) / 4.0)
            * (norm.ppf(prob) / (sharpe - target_sharpe)) ** 2
        )
        return min_track
    

    g_3 = skew(excess_returns)
    g_4 = kurtosis(excess_returns)

    res_dict['MIN_TRL_SRGE1_99.99%']='{0:.0f} bars or {1:.2f} years'.format(minTRL(sr,g_3,g_4,1,0.9999),minTRL(sr,g_3,g_4,1,0.9999)/252)
    
    res_dict['MIN_TRL_SRGE2_99.99%']='{0:.0f} bars or {1:.2f} years'.format(minTRL(sr,g_3,g_4,2,0.9999),minTRL(sr,g_3,g_4,2,0.9999)/252)
    
    res_dict['MIN_TRL_SRGE3_99.99%']='{0:.0f} bars or {1:.2f} years'.format(minTRL(sr,g_3,g_4,3,0.9999),minTRL(sr,g_3,g_4,3,0.9999)/252)
    

    
    
    #2. Average AUM
    avg_aum                             = np.nanmean(np.nansum(np.abs(bets.dollars_at_open),axis=1))
    res_dict['AVERAGE_AUM']             = hf.millions_fmt(avg_aum)
    
            
    #3. Capacity of Strategy
    
    res_dict['STRATEGY_CAPACITY']       = hf.millions_fmt(0)
    
    
    #4. Leverage (!!! Double check -something to do with sum of long_lev and short_lev > 1)
    avg_pos_size                        =  np.nanmean(np.nansum(bets.dollars_at_open,axis=1))
    res_dict['AVERAGE_POSITION_SIZE']   = hf.millions_fmt(avg_pos_size)
   

    res_dict['NET_LEVERAGE']            = '{0:.1f}'.format(avg_pos_size/avg_aum)
    
    
    #5. Turnover
    daily_shares        = np.nansum(bets.purchased_shares,axis=1)
    daily_value_traded  = np.nansum(np.abs(bets.dollars_at_open),axis=1)
    daily_turnover      = daily_shares/(2*daily_value_traded)
    
    res_dict['AVERAGE_DAILY_TURNOVER']  = hf.millions_fmt(np.mean(daily_turnover))
    
    #6. Correlation to underlying
    res_dict['CORRELATION_WITH_UNDERLYING'] = '{0:.2f}'.format(np.corrcoef(bets.underlying_daily_returns,bets.strategy_log_returns)[0,1])
    
    #7. Ratio of longs
    
    res_dict['LONG_RATIO'] = '{0:.2f} %'.format((((bets.cleaned_strategy_weights>0).sum())/(np.ones(bets.cleaned_strategy_weights.shape,dtype=bool).sum()))*100)
    
    #8. Maximum dollar position size 
    
    res_dict['MAX_SIZE'] = '{0:.2f} %'.format(np.nanmax(np.abs(bets.cleaned_strategy_weights))*100)
    
    #9. Stability of Wealth Process
    
    cum_log_returns = np.log1p(bets.strategy_log_returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]
    
    res_dict['STABILITY_OF_WEALTH_PROCESS']='{0:.2f} %'.format((rhat**2)*100)
    

    
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
    res_dict['PNL_FROM_STRATEGY'] = hf.millions_fmt(curves['Strategy'][-1])
    
    res_dict['PNL_FROM_LONG']    = hf.millions_fmt(curves['Long Contribution'][-1])
    
    #3. Annualized rate of return (Check this)
    res_dict['ANNUALIZED_MEAN_RETURN'] = '{0:.2f} %'.format((((1+np.mean(bets.strategy_daily_returns))**(365)-1))*100)
    
    res_dict['CUMMULATIVE_RETURN']= '{0:.2f} %'.format((np.cumprod(1+bets.strategy_daily_returns)[-1]-1)*100)
    
    yrs = int(len(cleaned_index))/252
    
    cagr_strategy = (((curves['Strategy'][-1]/curves['Strategy'][0])**(1/yrs))-1)
    
    res_dict['CAGR_STRATEGY'] = '{0:.2f} %'.format((((curves['Strategy'][-1]/curves['Strategy'][0])**(1/yrs))-1)*100)
   
    res_dict['CAGR_BENCHMARK'] = '{0:.2f} %'.format((((curves['Benchmark'][-1]/curves['Benchmark'][0])**(1/yrs))-1)*100)
    
    
    #4. Hit Ratio
    
    res_dict['HIT_RATIO'] = '{0:.2f} %'.format((((bets.daily_pnl>0).sum())/((bets.daily_pnl>0).sum()+(bets.daily_pnl<0).sum()+(bets.daily_pnl==0).sum()))*100)
    

    ##############################################################################################
    # C. Runs
    # 1. Runs concentration
    
    def runs(returns):
        wght=returns/returns.sum()
        hhi=(wght**2).sum()
        hhi=(hhi-returns.shape[0]**-1)/(1.-returns.shape[0]**-1)
        return hhi
    
    
    res_dict['HHI_PLUS'] = '{0:.5f}'.format(runs(bets.strategy_log_returns[bets.strategy_log_returns>0]))
    
    res_dict['HHI_MINUS'] = '{0:.5f}'.format(runs(bets.strategy_log_returns[bets.strategy_log_returns<0]))
    
    # 2. Drawdown and Time under Water

    
    
    def MDD(returns):
        def returns_to_dollars(amount,ret):
            return amount*np.cumprod(1+ret)
        
        doll_series = pd.Series(returns_to_dollars(100,returns))
        
        Roll_Max = doll_series.cummax()
        Daily_Drawdown = doll_series/Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        return Max_Daily_Drawdown
    
   
    DD_strategy=MDD(bets.strategy_log_returns)
    DD_benchmark=MDD(benchmark_returns)
    
    mdd_strat = DD_strategy.min()
    res_dict['MDD_STRATEGY']    = '{0:.2f} %'.format(DD_strategy.min()*100)
    res_dict['MDD_BENCHMARK']   = '{0:.2f} %'.format(DD_benchmark.min()*100)

    
    
    #3. 95 percentile
#    res_dict['95PERCENTILE_DRAWDOWN_STRATEGY']=DD_strategy.quantile(0.05)
#    res_dict['95PERCENTILE_DRAWDOWN_BENCHMARK']=DD_benchmark.quantile(0.05)

    
    #############################################################################################
    # D. Efficiency
    
    #1. Sharpe Ratio
    excess_returns = bets.strategy_log_returns-rf_returns
    
    res_dict['SHARPE_RATIO'] = '{0:.2f}'.format(mth.sqrt(252)*np.mean(excess_returns)/np.std(excess_returns))

    res_dict['PROBABILISTIC_SR'] = '{0:.2f} %'.format((norm.cdf(((sr-2)*mth.sqrt((len(excess_returns)-1)/252))/(mth.sqrt(1-(g_3*sr)+(0.25*(g_4-1)*sr*sr)))))*100)
        
    #2. sortino Ratio

    res_dict['SORTINO_RATIO'] = '{0:.2f}'.format(mth.sqrt(252)*np.mean(excess_returns)/np.std(excess_returns[excess_returns<np.mean(excess_returns)]))

    
    #2.Probabilistic Sharpe ratio

    #3.Information ratio
    excess_returns_benchmark = bets.strategy_log_returns-benchmark_returns
    res_dict['INFORMATION_RATIO'] = '{0:.2f}'.format(mth.sqrt(252)*np.mean(excess_returns_benchmark)/np.std(excess_returns_benchmark))
    
    #3. t_stat & P-value
    m = np.mean(excess_returns)
    s = np.std(excess_returns)
    n = len(excess_returns)
    t_stat = (m/s)*mth.sqrt(n)
    
    res_dict['t_STATISTIC']= '{0:.2f}'.format(t_stat)
    
    pval = stats.t.sf(np.abs(t_stat), n**2-1)*2 # Must be two-sided as we're looking at <> 0
    
    res_dict['p-VALUE']='{0:.5f} %'.format(pval*100)
    
    if pval <= 0.0001:
        res_dict['SIGNIFICANCE_AT_0.01%']='STATISTICALLY_SIGNIFICANT'
    else:
        res_dict['SIGNIFICANCE_AT_0.01%']='NOT_STATISTICALLY_SIGNIFICANT'
        
    #4. Omega Ratio 
    
    returns_less_thresh = excess_returns-(((100)**(1/252))-1)
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    res_dict['OMEGA_RATIO']='{0:.2f}'.format(numer/denom)
    
    #5. Tail Ratio
    
    res_dict['TAIL_RATIO']='{0:.2f}'.format(np.abs(np.percentile(bets.strategy_log_returns, 95)) /np.abs(np.percentile(bets.strategy_log_returns, 5)))
    
    
    #6. Rachev Ratio 
    left_threshold = np.percentile(excess_returns, 5)
    right_threshold = np.percentile(excess_returns, 95)
    CVAR_left = -1*(np.nanmean(excess_returns[excess_returns<=left_threshold]))
    CVAR_right = (np.nanmean(excess_returns[excess_returns>=right_threshold]))
    
    res_dict['RACHEV_RATIO']='{0:.2f}'.format(CVAR_right/CVAR_left)
    

    #############################################################################################
    # E. RISK MEASURES
    
    #1. SKEWNESS, KURTOSIS
    res_dict['SKEWNESS'] = '{0:.2f}'.format(stats.skew(bets.strategy_log_returns, bias = False))
    res_dict['KURTOSIS'] = '{0:.2f}'.format(stats.kurtosis(bets.strategy_log_returns, bias = False))
    
    #2. ANNUALIZED VOLATILITY
    res_dict['ANNUALIZED_VOLATILITY'] = '{0:.2f} %'.format(np.std(bets.strategy_log_returns)*np.sqrt(252)*100)

    #3. MAR Ratio
    res_dict['MAR_RATIO']= '{0:.2f}'.format((cagr_strategy)/abs(mdd_strat))
    
    #4 Tracking Error
    res_dict['TRACKING_ERROR']= '{0:.4f}'.format(np.std(bets.strategy_log_returns-benchmark_returns,ddof=1))
    
    percentile = 0.001
    res_dict['VaR_99.9']= '{0:.3f} %'.format(np.percentile(np.sort(bets.strategy_log_returns), percentile * 100)*100)
    
    
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
    
        
        return dfoutput,results_df
    
    _,res_dict['FACTOR_RES']= assetPriceReg(excess_returns, fama_factors)

   


    #############################################################################################
    # H. Bootstrap Stats
    # 1. Sharpe Bootstrap
    from arch.bootstrap import MovingBlockBootstrap
    from numpy.random import RandomState  
    
    def geom_mean(y):
        log_ret = np.log(1+y)
        geom = np.exp(np.sum(log_ret)/len(log_ret))-1
        return geom
    
    geo_avg = geom_mean(bets.strategy_daily_returns)
    detrended_ret = bets.strategy_daily_returns- geo_avg
    bs_sharpe = MovingBlockBootstrap(5,detrended_ret, random_state=RandomState(1234))
    
    

    res = bs_sharpe.apply(geom_mean,10000)      
    plots.density_plot_bootstrap(res,geo_avg)
    p_val = (res<=geo_avg).sum()/len(res)
    res_dict['GM_BOOTSTRAP_p_val']= '{0:.3f} %'.format(p_val*100)
    


############################################################################################





    return res_dict
    
