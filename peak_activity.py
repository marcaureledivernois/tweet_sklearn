import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import os
import datetime
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS
import seaborn as sns
import scipy
from scipy.stats import tiecorrect, rankdata, gaussian_kde
from pandas.tseries.offsets import BDay
from itertools import product
import seaborn as sns
from scipy.stats import t

#df = pd.read_pickle('df_clean.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_1.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_2.pkl')
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_new.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_new.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl')])  #sent with opti thresholds

#sentratio_and_price = pd.read_pickle('sentratio_and_price.pkl')

#=========================== volume of daily tweets across time ============================

daily_volume_total = pd.DataFrame(df[['date','sent_merged']].set_index('date').groupby(pd.Grouper(freq='D', offset = timedelta(hours=-8))).count())
weekly_volume_total = pd.DataFrame(df[['date','sent_merged']].set_index('date').groupby(pd.Grouper(freq='W')).count())

daily_volume_total = daily_volume_total.reset_index()
daily_volume_total.rename(columns={'sent_merged':'total_vol_tweet'}, inplace=True)
daily_volume_total['date'] = daily_volume_total['date'] + timedelta(days=1)
daily_volume_total['date'] = daily_volume_total['date'].apply(lambda x: x.normalize())

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dr = pd.date_range(start=df['date'].min(), end=df['date'].max())
dfh = pd.DataFrame()
dfh['Date'] = dr
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())
holidays = holidays.append(pd.DatetimeIndex([datetime(2020,3,23), datetime(2019,4,19), datetime(2018,12,5), datetime(2018,3,30),
                                             datetime(2017,4,14), datetime(2016,3,25), datetime(2015,4,3), datetime(2014,4,18),
                                             datetime(2013,3,29), datetime(2012,10,29), datetime(2012,10,30),
                                             datetime(2012,4,6), datetime(2011,4,22)]))  # manually add bank holidays (which are not official federal holidays)

isBusinessDay = BDay().onOffset
match_series = pd.to_datetime(daily_volume_total.date).map(isBusinessDay)
match_series.index = daily_volume_total.date
match_series = match_series & ~match_series.index.isin(holidays)
match_series_copy = pd.DataFrame(match_series)
match_series_copy.rename(columns={'date':'businessday'}, inplace=True)
match_series_copy = match_series_copy.reset_index()
match_series = match_series.reset_index(drop=True)

see = pd.DataFrame(data={'date': daily_volume_total.loc[match_series,'date'],'vol':daily_volume_total.loc[match_series,'total_vol_tweet']})

plt.plot(daily_volume_total.loc[match_series,'date'],daily_volume_total.loc[match_series,'total_vol_tweet'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Message volume (daily agg.)')
plt.show()

plt.plot(weekly_volume_total.index[:-1],weekly_volume_total.sent_merged[:-1])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Message volume (weekly agg.)')
plt.show()


#=========================== use classified tweets =========================================

#sentiment ratio aggregated on a daily basis VS daily return

def Nbullish(x):
    bull = x['sent_merged'] == 1
    return x.loc[bull, 'sent_merged'].count()

def Nbearish(x):
    bear = x['sent_merged'] == -1
    return x.loc[bear, 'sent_merged'].count()

def N_unclassif(x):
    uncl = x['sent'] == 0
    return x.loc[uncl, 'sent'].count()

def Nneutral(x):
    neut = x['sent_merged'] == 0
    return x.loc[neut, 'sent_merged'].count()

#=========================== empty dataframe =========================================

tickers = df['ticker'].unique().tolist()
uniques = [pd.date_range(start='2010-01-01', end='2020-03-23', freq='D').tolist(), tickers]   #todo start='2012-01-01', end='2020-03-23'
cadre = pd.DataFrame(product(*uniques), columns = ['date','ticker'])
cadre = cadre.sort_values(['ticker', 'date'], ascending=[1, 1])

#=========================== populate ================================================

sentratio = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D',offset = timedelta(hours=-8)),'ticker']).apply(Nbullish))
sentratio.rename(columns={0:'Nbullish'}, inplace=True)
sentratio['Nbearish'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D', offset = timedelta(hours=-8)),'ticker']).apply(Nbearish))
sentratio['N'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D', offset = timedelta(hours=-8)),'ticker'])['sent_merged'].count())['sent_merged']
sentratio['Polarity'] = (sentratio['Nbullish']-sentratio['Nbearish'])/(10+sentratio['Nbearish']+sentratio['Nbullish']) # I add 10 to the denom so polarity on days with few bullish and 0 bearish are close to 0
sentratio = sentratio.reset_index()

sentratio['date'] = sentratio['date'] + timedelta(days=1)
sentratio['date'] = sentratio['date'].apply(lambda x: x.normalize())

cadre['date'] = pd.to_datetime(cadre['date'], utc = True)
sentratio['date'] = pd.to_datetime(sentratio['date'], utc = True)
sentratio = pd.merge(cadre, sentratio,  how='left', left_on=['date','ticker'], right_on = ['date','ticker'])
sentratio = sentratio.fillna(0)

#============================== save =========================================================#
#dailpolarity3
#dailypolarity3_closetoclose

#with open('dailypolarity3_closetoclose.pkl', 'wb' ) as f:
#    pickle.dump(sentratio, f)

#============================== load =========================================================#

sentratio = pd.read_pickle('dailypolarity3_closetoclose.pkl')

N_byfirm = pd.DataFrame(sentratio[sentratio['date']>'2011-01-01'].groupby('ticker')['N'].median())

length = []
for thresh in range(100):
    threshold_bigfirm = thresh
    N_byfirm['big'] = N_byfirm['N']>threshold_bigfirm
    tickers_to_keep = list(N_byfirm[N_byfirm['big']].index)
    length.append(len(tickers_to_keep))
    print(len(tickers_to_keep))

plt.plot(range(100),length)
plt.xlabel('Threshold')
plt.ylabel('Number of firms')
plt.show()

threshold_bigfirm = 10     #to get 50firms, need threshold 16. #eventstudy uses threshold 50
N_byfirm['big'] = N_byfirm['N'] > threshold_bigfirm
tickers_to_keep = list(N_byfirm[N_byfirm['big']].index)
print('there are currently ' + str(tickers_to_keep.__len__()) + ' firms in the sample')

sentratio = sentratio[sentratio['ticker'].isin(tickers_to_keep)]
sentratio = sentratio.reset_index(drop=True)


#============================== merge sentiment and market var =========================================================#

matching_table = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'table_de_corr_changetickers.csv'),sep=';')

prices_db = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\tweet_sklearn\\daily_crsp.csv",sep=',')
prices_db.TICKER = prices_db.TICKER.replace(matching_table.set_index('old').new.to_dict()) #correct tickers with matching table
prices_db['date'] = pd.to_datetime(prices_db['date'],format="%Y%m%d")
prices_db['adjprice'] = np.abs(prices_db['PRC']/prices_db['CFACPR'])
prices_db = prices_db.sort_values(['TICKER', 'date'], ascending=[1, 1])
prices_db['daily_return'] = prices_db.groupby('TICKER').adjprice.pct_change()
#prices_db['daily_return'] = prices_db.groupby("TICKER")['adjprice'].apply(lambda x: np.log(x) - np.log(x.shift()))
prices_db['date'] = pd.to_datetime(prices_db['date'], utc = True)

sentratio_and_price = pd.merge(sentratio, prices_db,  how='left', left_on=['date','ticker'], right_on = ['date','TICKER'])

sentratio_and_price = sentratio_and_price.drop(['PERMNO','COMNAM', 'PERMCO','ISSUNO','CUSIP','BIDLO','ASKHI','PRC','RET','BID','ASK',
                                                'CFACPR','CFACSHR','RET','CFACPR','CFACSHR', 'TICKER','OPENPRC','NUMTRD'], axis=1)


#===============================  activity inside/outside trading days =====================================
# users post three times more during trading days than outside trading days
outside = sentratio_and_price[sentratio_and_price['daily_return'].isna()]['N'].mean()
inside = sentratio_and_price[~sentratio_and_price['daily_return'].isna()]['N'].mean()
print('avg posts inside/outside trading days : ', int(inside) ,'/', int(outside))

#===============================  define events =====================================

sentratio_and_price['medianN'] = sentratio_and_price.groupby('ticker')['N'].rolling(31, center = True).median().reset_index(0,drop=True)
sentratio_and_price['phi'] = (sentratio_and_price['N'] - sentratio_and_price['medianN']) / np.maximum(sentratio_and_price['medianN'],10)
sentratio_and_price['Event'] = sentratio_and_price['phi']  > 4
sentratio_and_price.loc[sentratio_and_price['date'] < '2011-02-01', ['Event']] = False

#===============================  merge with daily volume =====================================

sentratio_and_price = pd.merge(sentratio_and_price, daily_volume_total[['date','total_vol_tweet']],  how='left', left_on='date', right_on = 'date')
sentratio_and_price = pd.merge(sentratio_and_price, match_series_copy,  how='left', left_on='date', right_on = 'date')

#=============================== adjust polarity and N because of weekends =====================


sentratio_and_price['N_adj'] = np.nan
sentratio_and_price['Nbullish_adj'] = np.nan
sentratio_and_price['Nbearish_ajd'] = np.nan
sentratio_and_price['total_vol_tweet_ajd'] = np.nan

def adj(x):
    i, Nc, Nbullishc, Nbearishc, Ntotal = 0,0,0,0,0
    for i in x.index:
        if x.loc[i,'businessday']==False:
            Nc += x.loc[i,'N']
            Nbullishc += x.loc[i,'Nbullish']
            Nbearishc += x.loc[i,'Nbearish']
            Ntotal += x.loc[i,'total_vol_tweet']
        else:
            x.loc[i, 'N_adj'] = x.loc[i, 'N'] + Nc
            x.loc[i, 'Nbullish_adj'] = x.loc[i, 'Nbullish'] + Nbullishc
            x.loc[i, 'Nbearish_ajd'] = x.loc[i, 'Nbearish'] + Nbearishc
            x.loc[i, 'total_vol_tweet_ajd'] = x.loc[i, 'total_vol_tweet'] + Ntotal
            Nc, Nbullishc, Nbearishc, Ntotal = 0, 0, 0, 0
    return x

sentratio_and_price = sentratio_and_price.groupby('ticker').apply(adj)
sentratio_and_price = sentratio_and_price.drop(sentratio_and_price[sentratio_and_price['businessday']==False].index).reset_index(drop=True)

sentratio_and_price['Polarity'] = (sentratio_and_price['Nbullish_adj']-sentratio_and_price['Nbearish_ajd'])/\
                                      (10+sentratio_and_price['Nbearish_ajd']+sentratio_and_price['Nbullish_adj']) # I add 10 to the denom so polarity on days with few bullish and 0 bearish are close to 0
sentratio_and_price['N'] = sentratio_and_price['N_adj']
sentratio_and_price['total_vol_tweet'] = sentratio_and_price['total_vol_tweet_ajd']

#======================== load and save ================================================================================
# sentratio_and_price
# sentratio_and_price_closetoclose             #polarty computed from 4pm to 4pm
# sentratio_and_price_closetoclose_ajd         #adjusted for non business days. info trailed to next business day available
# sentratio_and_price_closetoclose_adj2.pkl    #with column adj_price (useful for portfolios later)
# sentratio_and_price_closetoclose_adj3.pkl    #correction of some negative prices. threshold=10


#with open('sentratio_and_price_closetoclose_adj3.pkl', 'wb' ) as f:
#    pickle.dump(sentratio_and_price, f)

sentratio_and_price = pd.read_pickle('sentratio_and_price_closetoclose_adj2.pkl')


#================================ define events - abnormal volume ============================

def mm_vol(tic,dat):
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == tic].copy()
    mask = (cut_tick['date'] > dat - timedelta(days=365)) & (cut_tick['date'] <= dat - timedelta(days=1)) & (cut_tick['businessday'] == True)
    cut_window = cut_tick.loc[mask]
    y = np.array(cut_window[['N']].diff())
    X = cut_window[['total_vol_tweet']].diff()
    X = np.array(sm.add_constant(X))
    if ~pd.DataFrame(y).isnull().all().values[0]:
        reg = sm.OLS(y,X,missing='drop').fit()
        par = reg.params
        r2 = reg.rsquared
    else:
        par = np.array([0, 1])
        r2 = 0
    return np.concatenate((par,r2),axis = None)

sentratio_and_price['mm_params_vol'] = sentratio_and_price.loc[sentratio_and_price['date']>'2011-01-01',:].apply(lambda row: mm_vol(row['ticker'], row['date']), axis=1)

def mm_vol_relative(tic,dat):
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == tic].copy()
    mask = (cut_tick['date'] > dat - timedelta(days=365)) & (cut_tick['date'] <= dat - timedelta(days=1)) & (cut_tick['businessday'] == True)
    cut_window = cut_tick.loc[mask]
    y = np.array(cut_window[['N']].pct_change())
    X = cut_window[['total_vol_tweet']].pct_change()
    X = np.array(sm.add_constant(X))
    if ~pd.DataFrame(y).isnull().all().values[0]:
        reg = sm.OLS(y,X,missing='drop').fit()
        par = reg.params
        r2 = reg.rsquared
    else:
        par = np.array([0, 1])
        r2 = 0
    return np.concatenate((par,r2),axis = None)

sentratio_and_price['mm_params_vol_relative'] = sentratio_and_price.loc[sentratio_and_price['date']>'2011-01-01',:].apply(lambda row: mm_vol_relative(row['ticker'], row['date']), axis=1)
sentratio_and_price.loc[sentratio_and_price['businessday']==True,'vol_tweet_changeperc'] = sentratio_and_price.loc[sentratio_and_price['businessday']==True,:].groupby('ticker').N.pct_change()
sentratio_and_price.loc[sentratio_and_price['businessday']==True,'market_vol_tweet_changeperc'] = sentratio_and_price.loc[sentratio_and_price['businessday']==True,:].groupby('ticker').total_vol_tweet.pct_change()

#================================== plot of volume relative change =====================================================

plt.scatter(sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'date'],
            sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'market_vol_tweet_changeperc'])
plt.xlabel('date')
plt.ylabel('Relative volume change')
plt.title('Market')
plt.show()

plt.scatter(sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'date'],
            sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'vol_tweet_changeperc'])
plt.xlabel('date')
plt.ylabel('Relative volume change')
plt.title('AAPL')
plt.show()

kde = gaussian_kde(sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'market_vol_tweet_changeperc'])
x = np.linspace(sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'market_vol_tweet_changeperc'].min(),
                sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'market_vol_tweet_changeperc'].max(), 100)
p = kde(x)
plt.plot(x, p)
plt.axvline(x=np.quantile(sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'market_vol_tweet_changeperc'], 0.33), color='red', linestyle='--')
plt.axvline(x=np.quantile(sentratio_and_price.loc[(sentratio_and_price['date']>'2011-01-01') & (sentratio_and_price['ticker']=='AAPL'),'market_vol_tweet_changeperc'], 0.66), color='red', linestyle='--')
plt.title('Empirical distribution of market volume relative change')
plt.show()


#============================== reg volume vs Number tweets posted =====================================================#

sentratio_and_price['transac_changeperc'] = sentratio_and_price.groupby(['ticker'])['VOL'].pct_change()

def reg_vol_N(data,N_mini,startyear):
    sentratio_and_price_copy = data[['date','ticker','vol_tweet_changeperc','transac_changeperc','N']].copy()
    with pd.option_context('mode.use_inf_as_na', True):
        sentratio_and_price_copy = sentratio_and_price_copy.dropna().reset_index(drop=True)

    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.date < pd.to_datetime(datetime(startyear, 1, 1), utc = True) ].index)
    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.N < N_mini].index)
    X = sm.add_constant(sentratio_and_price_copy[['vol_tweet_changeperc']])  # --- with constant

    y = sentratio_and_price_copy['transac_changeperc']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = OLS(yTrain, xTrain)
    res = model.fit()
    par = res.params

    xplot = sentratio_and_price_copy['vol_tweet_changeperc'].values
    yplot = sentratio_and_price_copy['transac_changeperc'].values
    plt.scatter(xplot,yplot, s=1)
    plt.plot(sentratio_and_price_copy['vol_tweet_changeperc'].values, par[1] * sentratio_and_price_copy['vol_tweet_changeperc'].values + par[0],
             label='y={:.2f}x+{:.2f}'.format(par[0], par[1]), color='red', linewidth=2)
    plt.xlabel('Relative change in number of tweets posted daily')
    plt.ylabel('Relative change in daily volume of transactions')
    plt.title('Activity vs Volume of Transaction')
    plt.xlim([-1, 5])
    plt.ylim([-1, 5])
    plt.grid(b='blue')
    plt.legend(loc='lower right')
    #plt.savefig('activity-volume.jpg')
    plt.show()
    print(res.summary())

reg_vol_N(sentratio_and_price,20,2012) #shows nicely that when ppl post more, the volume of transactions increases
#todo this motivates that ppl tweet on the day of the event


#################################### vol tweet daily ###########################################################

plt.plot(sentratio_and_price.loc[sentratio_and_price['ticker']=='AAPL','date'],sentratio_and_price.loc[sentratio_and_price['ticker']=='AAPL','total_vol_tweet'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Message volume (daily agg.)')
plt.show()

#===============================  standardize polarity =====================================

spy = sentratio_and_price[sentratio_and_price['ticker']=='SPY']
spy['date'] = pd.to_datetime(spy['date'], utc = True)
sentratio_and_price = pd.merge(sentratio_and_price, spy[['date','daily_return']],  how='left', left_on='date', right_on = 'date')
sentratio_and_price.rename(columns={'daily_return_x':'daily_return','daily_return_y':'market_return'}, inplace=True)

market_polarity = pd.read_pickle('market_polarity2_closetoclose_adj.pkl')
market_polarity['date'] = pd.to_datetime(market_polarity['date'], utc = True)
sentratio_and_price = pd.merge(sentratio_and_price, market_polarity[['date','market_polarity','spy_polarity']],  how='left', left_on='date', right_on = 'date')


def mm_ret(tic,dat):
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == tic].copy()
    mask = (cut_tick['date'] > dat - timedelta(days=365)) & (cut_tick['date'] <= dat - timedelta(days=1))
    cut_window = cut_tick.loc[mask]
    y = cut_window[['daily_return']].values
    X = cut_window[['market_return']].values
    X = sm.add_constant(X)
    if ~pd.DataFrame(y).isnull().all().values[0]:
        par = sm.OLS(y,X,missing='drop').fit().params
    else:
        par = np.array([0, 1])
    return par

def mm_pol(tic,dat):
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == tic].copy()
    mask = (cut_tick['date'] > dat - timedelta(days=365)) & (cut_tick['date'] <= dat - timedelta(days=1))
    cut_window = cut_tick.loc[mask]
    y = cut_window[['Polarity']].values
    X = cut_window[['market_polarity']].values    #change this for spy_polarity
    X = sm.add_constant(X)
    if ~pd.DataFrame(y).isnull().all().values[0]:
        par = sm.OLS(y,X,missing='drop').fit().params
    else:
        par = np.array([0, 1])
    return par

def cm_pol(tic,dat):
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == tic].copy()
    mask = (cut_tick['date'] > dat - timedelta(days=365)) & (cut_tick['date'] <= dat - timedelta(days=1))
    cut_window = cut_tick.loc[mask]
    return cut_window['Polarity'].mean()

sentratio_and_price['mm_params_ret'] = sentratio_and_price.loc[sentratio_and_price['date']>'2011-01-01',:].apply(lambda row: mm_ret(row['ticker'], row['date']), axis=1)

sentratio_and_price['mm_params_pol'] = sentratio_and_price.loc[sentratio_and_price['date']>'2011-01-01',:].apply(lambda row: mm_pol(row['ticker'], row['date']), axis=1)
sentratio_and_price['cm_normal_pol'] = sentratio_and_price.loc[sentratio_and_price['date']>'2011-01-01',:].apply(lambda row: cm_pol(row['ticker'], row['date']), axis=1)

sentratio_and_price = sentratio_and_price[sentratio_and_price['date'] > '2011-01-01']

sentratio_and_price['normal_return'] = sentratio_and_price['mm_params_ret'].map(lambda x: x[0]) + sentratio_and_price['market_return'] * sentratio_and_price.mm_params_ret.map(lambda x: x[1])
sentratio_and_price['mm_normal_pol'] = sentratio_and_price['mm_params_pol'].map(lambda x: x[0]) + sentratio_and_price['market_polarity'] * sentratio_and_price.mm_params_pol.map(lambda x: x[1])

sentratio_and_price['abnormal_return'] = sentratio_and_price['daily_return'] - sentratio_and_price['normal_return']
#sentratio_and_price['abnormal_pol'] = sentratio_and_price['Polarity'] - sentratio_and_price['cm_normal_pol']   #chose this for constant mean normal polarity
sentratio_and_price['abnormal_pol'] = sentratio_and_price['Polarity'] - sentratio_and_price['mm_normal_pol']  #chose this for market model normal polarity

sentratio_and_price['mm_normal_vol_delta'] = sentratio_and_price['mm_params_vol'].map(lambda x: x[0]) + np.array(sentratio_and_price.groupby('ticker')['total_vol_tweet'].diff()) * sentratio_and_price.mm_params_vol.map(lambda x: x[1])
sentratio_and_price['mm_normal_vol_delta'] = sentratio_and_price['mm_normal_vol_delta'].clip(0, None)
sentratio_and_price['N_t-1'] = sentratio_and_price.groupby('ticker')['N'].shift(1)
sentratio_and_price['mm_normal_vol'] = sentratio_and_price['N_t-1'] + sentratio_and_price['mm_normal_vol_delta']
sentratio_and_price['abnormal_vol'] = sentratio_and_price['N'] - sentratio_and_price['mm_normal_vol']

sentratio_and_price['mm_normal_vol_delta_relative'] = sentratio_and_price['mm_params_vol_relative'].map(lambda x: x[0]) + \
                                                      np.array(sentratio_and_price['market_vol_tweet_changeperc']) * sentratio_and_price.mm_params_vol_relative.map(lambda x: x[1])

sentratio_and_price['abnormal_vol_relative'] = sentratio_and_price['vol_tweet_changeperc'] - sentratio_and_price['mm_normal_vol_delta_relative']


sentratio_and_price['phi2'] = sentratio_and_price[['ticker','abnormal_vol']].groupby('ticker').transform(lambda x: (x - x.mean()) / x.std())
sentratio_and_price['Event2'] = sentratio_and_price['phi2']  > 2.2   #2.2   3


sentratio_and_price['abnormal_vol_relative'] = sentratio_and_price['abnormal_vol_relative'].replace([np.inf, -np.inf], np.nan)
sentratio_and_price['phi3'] = sentratio_and_price[['ticker','abnormal_vol_relative']].groupby('ticker').transform(lambda x: (x - x.mean()) / x.std())
sentratio_and_price['Event3'] = sentratio_and_price['phi3']  > 2   #2.2   3



sentratio_and_price.loc[sentratio_and_price['date'] < '2011-02-01', ['Event2']] = False
sentratio_and_price.loc[sentratio_and_price['date'] < '2011-02-01', ['Event3']] = False

#index1 = sentratio_and_price[(sentratio_and_price['Event3'] * sentratio_and_price['Event3'].shift(1)) == 1].index
#index2 = sentratio_and_price[(sentratio_and_price['Event3'] * sentratio_and_price['Event3'].shift(-1)) == 1].index
#sentratio_and_price.loc[index1,'Event3'] = False     #correct consecutive events
#sentratio_and_price.loc[index2,'Event3'] = False     #correct consecutive events
#sentratio_and_price.loc[sentratio_and_price['date'] > '2019-12-31', ['Event3']] = False #update price database
#===============================  create eventlist =====================================

sentratio_and_price['d'] = np.nan

events_index = sentratio_and_price[sentratio_and_price['Event3']].index
eventlist = []
for e in events_index:
    eventlist.append([sentratio_and_price.loc[e,'date'],sentratio_and_price.loc[e,'ticker'], e])
eventlist = pd.DataFrame(eventlist, columns = ['date','ticker','Index'])



#===============================  distribution polarities of events =====================================
from sklearn.neighbors import KernelDensity

abn_pol_events = sentratio_and_price.loc[(sentratio_and_price['Event3'] == True),'abnormal_pol']
abn_pol_all = sentratio_and_price['abnormal_pol']

run = 'no'
if run == 'yes':
    kde = gaussian_kde(abn_pol_events)
    x = np.linspace(abn_pol_events.min(), abn_pol_events.max(), 100)
    p = kde(x)
    plt.plot(x,p)
    plt.axvline(x=np.quantile(abn_pol_events,0.33), color = 'red',  linestyle='--')
    plt.axvline(x=np.quantile(abn_pol_events,0.66), color = 'red',  linestyle='--')
    plt.title('Empirical distribution of abnormal polarities at event dates')
    plt.show()

    plt.hist(abn_pol_events,bins='auto')
    plt.title('Histogram of abnormal polarities at event dates')
    plt.show()


    kde = gaussian_kde(abn_pol_all)
    x = np.linspace(abn_pol_all.min(), abn_pol_all.max(), 100)
    p = kde(x)
    plt.plot(x,p)
    plt.axvline(x=np.quantile(abn_pol_all,0.33), color = 'red',  linestyle='--')
    plt.axvline(x=np.quantile(abn_pol_all,0.66), color = 'red',  linestyle='--')
    plt.title('Empirical distribution of abnormal polarities')
    plt.show()

    hist = plt.hist(abn_pol_all,bins='auto')
    plt.title('Histogram of abnormal polarities')
    plt.show()

low_thresh = np.quantile(abn_pol_all,1/3)    #abn_pol_events
up_thresh = np.quantile(abn_pol_all,2/3)

#low_thresh = -0.08        #-0.08
#up_thresh = 0.14          #0.14

#===============================  define type of events =====================================

sentratio_and_price['Type'] = np.nan
sentratio_and_price.loc[(sentratio_and_price['Event3'] == True) & (sentratio_and_price['abnormal_pol']>up_thresh),'Type'] = 'Good'
sentratio_and_price.loc[(sentratio_and_price['Event3'] == True) & (sentratio_and_price['abnormal_pol']<low_thresh),'Type'] = 'Bad'
sentratio_and_price.loc[(sentratio_and_price['Event3'] == True) & (((sentratio_and_price['abnormal_pol'])<up_thresh) & ((sentratio_and_price['abnormal_pol'])>low_thresh)),'Type'] = 'Neutral'

print('# good/neutral/bad events: ' , np.sum(sentratio_and_price['Type']=='Good'),
      ' / ', np.sum(sentratio_and_price['Type']=='Neutral'), ' / ', np.sum(sentratio_and_price['Type']=='Bad'))
# good/neutral/bad events:  240  /  232  /  233

eventlist = pd.merge(eventlist[['date','ticker','Index']], sentratio_and_price[['date','ticker','Type']],  how='left', left_on=['date','ticker'], right_on = ['date','ticker'])



#===============================  event study =====================================


eventstudy = pd.DataFrame()
for _, row in eventlist.iterrows():
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == row['ticker']].copy()
    cut_window = cut_tick.loc[row['Index']-20:row['Index']+20].copy()
    if cut_window.shape[0] == 41:
        cut_window['d'] = np.linspace(-20,20,41)
        cut_window['daily_return'] = cut_window['daily_return'].fillna(0)
        cut_window['cum_abn_Pol'] = cut_window['abnormal_pol'].cumsum()
        cut_window['cumRet'] = cut_window['daily_return'].cumsum()
        cut_window['CAR'] = cut_window['abnormal_return'].cumsum()
        cut_window['CAR'] = cut_window['CAR'].fillna(method='ffill')
        cut_window['Type'] = row['Type']
        cut_window['Ticker'] = row['ticker']
        cut_window['EventDate'] = row['date']
        eventstudy = pd.concat([eventstudy, cut_window[['d','cumRet','cum_abn_Pol','CAR','Type','Ticker','EventDate']]])

sd = []
for d in range(-20,21):
    sdcar = np.sqrt(np.sum((eventstudy[eventstudy['d']==d]['CAR'] - eventstudy[eventstudy['d']==d].mean()['CAR'])**2)/eventlist.shape[0]**2)
    sdret = np.sqrt(np.sum((eventstudy[eventstudy['d']==d]['cumRet'] - eventstudy[eventstudy['d']==d].mean()['cumRet'])**2)/eventlist.shape[0]**2)
    sdpol = np.sqrt(np.sum((eventstudy[eventstudy['d']==d]['cum_abn_Pol'] - eventstudy[eventstudy['d']==d].mean()['cum_abn_Pol'])**2)/eventlist.shape[0]**2)
    sd.append([sdpol,sdret,sdcar])

sd = pd.DataFrame(data=sd,index=range(-20,21))
sd.rename(columns={0:'sd_cum_abn_pol',1:'sd_cumRet',2:'sd_CAR'}, inplace=True)


def event_study_grp(type):
    df = eventstudy[eventstudy['Type']==type][['d','cumRet','cum_abn_Pol','CAR']].groupby('d').mean(numeric_only=True)
    df = pd.merge(df, sd,  left_index=True, right_index = True)
    df['t_pol'] = df['cum_abn_Pol']/df['sd_cum_abn_pol']
    df['t_Ret'] = df['cumRet'] / df['sd_cumRet']
    df['t_CAR'] = df['CAR'] / df['sd_CAR']
    return df



neutral = event_study_grp('Neutral')
good = event_study_grp('Good')
bad = event_study_grp('Bad')


plt.plot(good.index,good['CAR'],'g--')
plt.plot(neutral.index,neutral['CAR'],'gray')
plt.plot(bad.index,bad['CAR'],'r-.')
plt.fill_between(bad.reset_index().d,bad['CAR']-2*bad['sd_CAR'],bad['CAR']+2*bad['sd_CAR'],facecolor='red', alpha = 0.1)
plt.fill_between(neutral.reset_index().d,neutral['CAR']-2*neutral['sd_CAR'],neutral['CAR']+2*neutral['sd_CAR'],facecolor='gray', alpha = 0.1)
plt.fill_between(good.reset_index().d,good['CAR']-2*good['sd_CAR'],good['CAR']+2*good['sd_CAR'],facecolor='green', alpha = 0.1)
plt.xlabel('Days from event')
plt.ylabel('CAAR')
plt.grid(b=None)
plt.tick_params(axis='both', labelsize=10, length = 3)
plt.legend(('Bullish', 'Neutral','Bearish'),
           loc='upper left')
#plt.savefig('ES_CAAR.jpg')
plt.show()

plt.plot(good.index,good['cum_abn_Pol'],'g--')
plt.plot(neutral.index,neutral['cum_abn_Pol'],'gray')
plt.plot(bad.index,bad['cum_abn_Pol'],'r-.')
plt.fill_between(bad.reset_index().d,bad['cum_abn_Pol']-2*bad['sd_cum_abn_pol'],bad['cum_abn_Pol']+2*bad['sd_cum_abn_pol'],facecolor='red', alpha = 0.1)
plt.fill_between(neutral.reset_index().d,neutral['cum_abn_Pol']-2*neutral['sd_cum_abn_pol'],neutral['cum_abn_Pol']+2*neutral['sd_cum_abn_pol'],facecolor='gray', alpha = 0.1)
plt.fill_between(good.reset_index().d,good['cum_abn_Pol']-2*good['sd_cum_abn_pol'],good['cum_abn_Pol']+2*good['sd_cum_abn_pol'],facecolor='green', alpha = 0.1)
plt.xlabel('Days from event')
plt.ylabel('CAAP')
plt.grid(b=None)
plt.tick_params(axis='both', labelsize=10, length = 3)
plt.legend(('Bullish', 'Neutral','Bearish'),
           loc='upper left')
#plt.savefig('ES_CAAP.jpg')
plt.show()

#todo very ncie plots. it looks like polarity trend is more persistent. people are biased towards the past instead of looking for the future.
#todo return: consistent with previous literature and EMH : adjust quickly

#===============================  plots =====================================




eventlist = pd.merge(eventlist[['date','ticker','Index','Type']], sentratio_and_price[['date', 'ticker', 'N',]], how='left', left_on=['ticker', 'date'],
                     right_on=['ticker', 'date'])

def plot_activity(ticker):
    cut_act = sentratio_and_price[(sentratio_and_price['ticker']==ticker) & (sentratio_and_price['businessday']==True)]
    cut_event = eventlist[eventlist['ticker']==ticker]

    plt.plot(cut_act['date'],cut_act['N'],'black')
    plt.scatter(cut_event.loc[cut_event['Type']=='Good','date'],cut_event.loc[cut_event['Type']=='Good','N'],marker='^', s=80, facecolors='none', edgecolors='g')
    plt.scatter(cut_event.loc[cut_event['Type'] == 'Neutral', 'date'], cut_event.loc[cut_event['Type'] == 'Neutral', 'N'],
                s=80, facecolors='none', edgecolors='gray')
    plt.scatter(cut_event.loc[cut_event['Type'] == 'Bad', 'date'], cut_event.loc[cut_event['Type'] == 'Bad', 'N'],
                s=80, facecolors='none',marker='v', edgecolors='r')
    plt.xlabel('Date')
    plt.ylabel('N')
    plt.title('Activity - ' + ticker)
    #plt.savefig('Activity_' + ticker + '.jpg')
    plt.show()

plot_activity('AAPL')

#=============================== events across time ================================

types = ['Bad','Neutral','Good']
uniques = [pd.date_range(start='2011-02-11', end='2020-02-26', freq='M').tolist(), types]
cadre = pd.DataFrame(product(*uniques), columns = ['date','Type'])
cadre.rename(columns={0:'date'}, inplace=True)
cadre = cadre.sort_values(['Type', 'date'], ascending=[1, 1])
cadre['date'] = pd.to_datetime(cadre['date'], utc = True)

events_per_day = pd.DataFrame(eventlist[['date','Type','Index']].set_index('date').groupby([pd.Grouper(freq='M'), 'Type']).count()).reset_index()
events_per_day = pd.merge(cadre, events_per_day,  how='left', left_on=['date','Type'], right_on = ['date','Type'])
events_per_day['Index'] = events_per_day['Index'].fillna(0)

events_pivot = events_per_day.pivot_table('Index', 'date', 'Type')

plt.stackplot(events_pivot.index,events_pivot.Bad,  events_pivot.Neutral, events_pivot.Good, labels=['Bearish','Neutral','Bullish'], colors = ['red','lightgray','green'])
plt.legend(reversed(plt.legend().legendHandles), reversed(['Bearish','Neutral','Bullish']))
plt.title('Number of events across time (monthly agg.)')
plt.show()


#===============================  correlations =====================================
#todo correlation are very high! improvement wrt previous literature (i.e RANCO2015)

corr_pol_ret = sentratio_and_price[['ticker','Polarity','daily_return']].groupby(
    'ticker').corr().iloc[0::2,-1].reset_index()[['ticker','daily_return']].rename(columns={'daily_return':'corr'})

corr_pol_abnret = sentratio_and_price[['ticker','Polarity','abnormal_return']].groupby(
    'ticker').corr().iloc[0::2,-1].reset_index()[['ticker','abnormal_return']].rename(columns={'abnormal_return':'corr'})

#===============================  tweets length during event days / outside event days =====================================

tweets_events = pd.DataFrame()
for _, row in eventlist.iterrows():
    cut_tick = df[(df['ticker'] == row['ticker']) & (df['date'] == row['date'])].copy()
    tweets_events = pd.concat([tweets_events, cut_tick[['ticker','id_msg','date','sent_merged','clean_text']]])

f = lambda x: len(x["clean_text"].split())
tweets_events['len'] = tweets_events.apply(f, axis=1)

tweets_noevents = df[~df.index.isin(tweets_events.index)]
tweets_noevents['len'] = tweets_noevents.apply(f, axis=1)

tweets_events['len'].describe() #mean     1.056665e+01 std      8.046778e+00
tweets_noevents['len'].describe() # mean     1.084647e+01 std      9.720390e+00

#===============================  regressions polarity / next day return =====================================

def regress(subset,Xvar,Yvar,lag=0):
    if(subset==None):
        sentratio_and_price_reg = sentratio_and_price.dropna(subset=['daily_return']).copy().reset_index(drop=True)
    else:
        sentratio_and_price_reg = sentratio_and_price[sentratio_and_price['ticker'].isin(subset)].dropna(subset=['daily_return']).copy().reset_index(drop=True)
    sentratio_and_price_reg = sentratio_and_price_reg[sentratio_and_price_reg['daily_return'].between(sentratio_and_price_reg['daily_return'].quantile(.05),
                                                                 sentratio_and_price_reg['daily_return'].quantile(.95))]  # without outliers
    sentratio_and_price_reg['daily_return_t+1'] = sentratio_and_price_reg.groupby(['ticker'])['daily_return'].shift(-1)
    sentratio_and_price_reg['abn_return_t+1'] = sentratio_and_price_reg.groupby(['ticker'])['abnormal_return'].shift(-1)
    sentratio_and_price_reg['CAP'] = sentratio_and_price_reg.groupby(['ticker'])['abnormal_pol'].transform(lambda x: x.rolling(lag, min_periods=lag).sum())
    sentratio_and_price_reg = sentratio_and_price_reg.dropna(subset=['daily_return_t+1','abn_return_t+1','abnormal_pol','CAP']).reset_index(drop=True)

    X = sm.add_constant(sentratio_and_price_reg[Xvar])  # --- with constant
    y = sentratio_and_price_reg[Yvar]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.001, random_state=1)
    model = OLS(yTrain, xTrain)
    res = model.fit()
    print(sentratio_and_price_reg.ticker.unique(), xTrain.shape[0])
    print(res.summary())

regress(None,['abnormal_pol'],'daily_return')

regress(None,['abnormal_pol'],'abnormal_return')   #polarity signif
regress(None,['abnormal_pol'],'abn_return_t+1')    #polarity non signif

regress(None,['abnormal_pol'],'daily_return')      #polarity signif
regress(None,['abnormal_pol'],'daily_return_t+1')  #polarity non signif

regress(None,['CAP'],'daily_return', lag=5)        #lags of CAP signif on rt, but not on r_t+1. Also, adding more lags decrease signif (as CAP becomes more noisy)
regress(None,['CAP'],'daily_return', lag=10)
regress(None,['CAP'],'daily_return', lag=20)

regress(None,['CAP'],'daily_return_t+1', lag=5)
regress(None,['CAP'],'daily_return_t+1', lag=10)
regress(None,['CAP'],'daily_return_t+1', lag=20)

regress(None,['Polarity'],'daily_return')   #polarity signif
regress(None,['Polarity'],'daily_return_t+1')    #polarity non signif


#todo polarity is signif when predicting contemporaneous abnormal return, but not next period abnormal return
#todo : story: in general cant predict next period return but around specific events it can

#===============================  boxplots =====================================


def test_distrib(full_db, day, var):
    full_db_copy = full_db[full_db['d'] == day].copy()
    full_db_copy = full_db_copy[full_db_copy['CAR'].between(full_db_copy['CAR'].quantile(.03),
                                                                 full_db_copy['CAR'].quantile(
                                                                     .97))]  # without outliers
    my_pal = {"Good": "green", "Neutral": "gray", "Bad": "red"}
    ax = sns.boxplot(x="Type", y=var, data=full_db_copy, whis=1.5, palette=my_pal,  order=["Good", "Neutral", "Bad"])
    #ax = sns.stripplot(x='Type', y=var, data=full_db_copy, color="orange", jitter=0.03, size=1.5)
    plt.title('day: ' + str(day))
    if var=='cum_abn_Pol':
        plt.ylabel('CAP')
    else:
        plt.ylabel(var)
    plt.xlabel("Type")
    plt.xticks([0,1,2], ['Bullish', 'Neutral', 'Bearish'])
    #plt.savefig('boxplot_' + var + str(day) +'.jpg')
    plt.show()

test_distrib(eventstudy, -5, 'cum_abn_Pol')
test_distrib(eventstudy, -5, 'CAR')

test_distrib(eventstudy, 0, 'cum_abn_Pol')
test_distrib(eventstudy, 0, 'CAR')

test_distrib(eventstudy, 5, 'cum_abn_Pol')
test_distrib(eventstudy, 5, 'CAR')

#===============================  mann whitney test =====================================

def MannWhitneyTest(data, day, type, var):
    ## H0 : theta_good = theta_neutral = theta_bad
    ## H0: theta_good > theta_neutral > theta_bad
    dat =  data[data['d'] == day].copy()

    U = scipy.stats.mannwhitneyu(dat[dat['Type'] == type[0]][var],
                                 dat[dat['Type'] == type[1]][var], use_continuity=False)[0]
    print(scipy.stats.mannwhitneyu(dat[dat['Type'] == type[0]][var],
                                 dat[dat['Type'] == type[1]][var], use_continuity=False))
    n1 = len(dat[dat['Type'] == type[0]][var])
    n2 = len(dat[dat['Type'] == type[1]][var])
    Z = (int(U) - int(n1) * int(n2) / 2) / np.sqrt(int(n1) * int(n2) * (int(n1) + int(n2) + 1) / 12)

    if Z<-1.9:
        concl = 'reject'
    else:
        concl = 'not reject'
    print('at t' + str(day) + ', we can ' + str(concl) + ' H0 : theta_' + type[0] + ' =' + ' theta_' + type[1] + ', where theta is the median of ' + str(var) + '. U=' + str(U) + ', Z=' + str(Z) + ', n1=' + str(n1) + ', n2=' + str(n2))
    return U, Z, n1, n2

Z = -1.1
n1 = 240
n2 = 232
print(Z * np.sqrt(n1*n2*(n1+n2+1)/12) + 0.5*n1*n2)

MannWhitneyTest(eventstudy, -5, ['Good','Neutral'], 'CAR')
MannWhitneyTest(eventstudy, -5, ['Neutral','Bad'], 'CAR')
#MannWhitneyTest(eventstudy, -5, ['Good','Bad'], 'CAR')
MannWhitneyTest(eventstudy, 0, ['Good','Neutral'], 'CAR')
MannWhitneyTest(eventstudy, 0, ['Neutral','Bad'], 'CAR')
MannWhitneyTest(eventstudy, 5, ['Good','Neutral'], 'CAR')
MannWhitneyTest(eventstudy, 5, ['Neutral','Bad'], 'CAR')

MannWhitneyTest(eventstudy, -5, ['Good','Neutral'], 'cum_abn_Pol')
MannWhitneyTest(eventstudy, -5, ['Neutral','Bad'], 'cum_abn_Pol')
MannWhitneyTest(eventstudy, 0, ['Good','Neutral'], 'cum_abn_Pol')
MannWhitneyTest(eventstudy, 0, ['Neutral','Bad'], 'cum_abn_Pol')
MannWhitneyTest(eventstudy, 5, ['Good','Neutral'], 'cum_abn_Pol')
MannWhitneyTest(eventstudy, 5, ['Neutral','Bad'], 'cum_abn_Pol')

#===============================  plots polarity vs return =====================================

def plot_bullishness_return_company(ticker):
    df = sentratio_and_price[sentratio_and_price['ticker']==ticker]
    df = df[df['daily_return'].between(df['daily_return'].quantile(.02),df['daily_return'].quantile(.98))]  # without outliers
    df = df[df['date'] > '2019-01-01']
    plt.style.use('seaborn-white')
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Polarity', color=color)
    #ax1.plot(df['date'], df['Bullishness'].shift(1), color=color)
    ax1.plot(df['date'], df['Polarity'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Return', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df['daily_return'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    print('Corr coef : ', df.corr()['Polarity']['daily_return'])
    plt.title(ticker + ' - Correlation between Polarity and Return : ' + "{0:.2f}".format(df.corr()['Polarity']['daily_return']))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    #fig.savefig(ticker + ' - 3M Bullishness vs Return.jpg')

plot_bullishness_return_company('AAPL')
plot_bullishness_return_company('AMZN')
plot_bullishness_return_company('TSLA')
plot_bullishness_return_company('AMD')
plot_bullishness_return_company('FB')
plot_bullishness_return_company('SPY')


#===============================  plots polarity vs return - LAGGED =====================================

def plot_bullishness_return_company_lagged(ticker):
    df = sentratio_and_price[sentratio_and_price['ticker']==ticker]
    df['daily_return_shift'] = df['daily_return'].shift(-1)
    df = df[df['daily_return'].between(df['daily_return'].quantile(.02),df['daily_return'].quantile(.98))]  # without outliers
    df = df[df['date'] > '2019-01-01']
    plt.style.use('seaborn-white')
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Polarity', color=color)
    ax1.plot(df['date'], df['Polarity'].shift(1), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Return t+1', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df['daily_return'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    print('Corr coef : ', df.corr()['Polarity']['daily_return_shift'])
    plt.title(ticker + ' - Correlation between Polarity and next-day Return : ' + "{0:.2f}".format(df.corr()['Polarity']['daily_return_shift']))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    #fig.savefig(ticker + ' - 3M Bullishness vs Return.jpg')

plot_bullishness_return_company_lagged('AAPL')
plot_bullishness_return_company_lagged('AMZN')
plot_bullishness_return_company_lagged('TSLA')
plot_bullishness_return_company_lagged('AMD')
plot_bullishness_return_company_lagged('FB')
plot_bullishness_return_company_lagged('SPY')



#===============================  plots unlabeled vs neutral =====================================

def N_lab_bull(x):
    bull = x['sent'] == 1
    return x.loc[bull, 'sent'].count()

def N_lab_bear(x):
    bear = x['sent'] == -1
    return x.loc[bear, 'sent'].count()



ts_neut = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='M')]).apply(Nneutral))
ts_neut.rename(columns={0:'N_neut_class'}, inplace=True)
ts_neut['N_bull_class'] =pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='M')]).apply(Nbullish))
ts_neut['N_bear_class'] =pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='M')]).apply(Nbearish))
ts_neut['N_unlabeled'] =pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='M')]).apply(N_unclassif))
ts_neut['N_lab_bull'] =pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='M')]).apply(N_lab_bull))
ts_neut['N_lab_bear'] =pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='M')]).apply(N_lab_bear))
ts_neut['N1'] = ts_neut['N_bull_class'] + ts_neut['N_bear_class'] + ts_neut['N_neut_class']
ts_neut['N2'] = ts_neut['N_lab_bull'] + ts_neut['N_lab_bear'] + ts_neut['N_unlabeled']

plt.stackplot(ts_neut.index,ts_neut['N_bear_class']/ts_neut['N1']*100,  ts_neut['N_neut_class']/ts_neut['N1']*100, ts_neut['N_bull_class']/ts_neut['N1']*100
              , labels=['% classified as bearish', '% classified as neutral','% classified as bullish'], colors = ['red','gray','green'])
plt.plot(ts_neut.index,ts_neut['N_lab_bear']/ts_neut['N1']*100,color='black')
plt.plot(ts_neut.index,(ts_neut['N_lab_bear']+ts_neut['N_unlabeled'])/ts_neut['N1']*100,color='black')
plt.legend(reversed(plt.legend().legendHandles), reversed(['% classified as bearish', '% classified as neutral','% classified as bullish']), frameon=True, loc = 'center left')
plt.title('Messages classification across time (month agg.)')
plt.show()


plt.stackplot(ts_neut.index,ts_neut['N_lab_bear']/ts_neut['N1']*100,
              (ts_neut['N_bear_class']-ts_neut['N_lab_bear'])/ts_neut['N1']*100,
              (ts_neut['N_neut_class'])/ts_neut['N1']*100,
              (ts_neut['N_bull_class']-ts_neut['N_lab_bull'])/ts_neut['N1']*100,
              (ts_neut['N_lab_bull'])/ts_neut['N1']*100,
             labels=['% labeled as bearish', '% classified as bearish','% classified as neutral','% classified as bullish', '% labeled as bullish'], colors = [(1, 0, 0, 1),(1, 0, 0, 0.4),'lightgray',(0.0, 0.5019607843137255, 0.0,0.4),(0.0, 0.5019607843137255, 0.0,1)])
plt.legend(reversed(plt.legend().legendHandles), reversed(['% labeled as bearish', '% classified as bearish','% classified as neutral','% classified as bullish', '% labeled as bullish']), frameon=True, loc = 'center left')
plt.title('Message classification across time (month agg.)')
plt.show()



plt.stackplot(ts_neut.index,ts_neut['N_lab_bear']/ts_neut['N1']*100,  ts_neut['N_unlabeled']/ts_neut['N1']*100, ts_neut['N_lab_bull']/ts_neut['N1']*100
              , labels=['% labeled as bearish', '% unlabeled','% labeled as bullish'], colors = ['red','lightgray','green'])
plt.legend(reversed(plt.legend().legendHandles), reversed(['% labeled as bearish', '% unlabeled','% labeled as bullish']), frameon=True, loc = 'center left')
plt.title('User-labeled messages across time (month agg.)')
plt.show()


plt.plot(ts_neut.index,ts_neut['N_bull_class']/ts_neut['N1'],'g')
plt.plot(ts_neut.index,ts_neut['N_neut_class']/ts_neut['N1'],'b')
plt.plot(ts_neut.index,ts_neut['N_bear_class']/ts_neut['N1'],'r')
plt.legend(('% classified as bullish', '% classified as neutral', '% classified as bearish'),loc='center right')
plt.title('Tweets classification across time (month agg.)')
plt.show()

plt.plot(ts_neut.index,ts_neut['N_lab_bull']/ts_neut['N1'],'g')
plt.plot(ts_neut.index,ts_neut['N_unlabeled']/ts_neut['N1'],'b')
plt.plot(ts_neut.index,ts_neut['N_lab_bear']/ts_neut['N1'],'r')
plt.legend(('% labeled as bullish', '% unlabeled', '% labeled as bearish'),loc='upper right')
plt.title('User-labeled tweets across time (month agg.)')
plt.show()


ts_neut['prop_unlabeled'] = ts_neut['N_uncl']/ts_neut['N']
ts_neut['prop_class_neutral'] = ts_neut[0]/ts_neut['N']

plt.plot(ts_neut.index,ts_neut['prop_unlabeled'],'r')
plt.plot(ts_neut.index,ts_neut['prop_class_neutral'],'b')
plt.legend(('% Unlabeled', '% Classified as neutral'),loc='upper right')
plt.title('Unlabeled tweets and neutral class')
plt.show()


#################################### number of firms per day ###########################################################

nunique_aftertrim = pd.DataFrame(sentratio_and_price[['date','ticker']].groupby('date')['ticker'].nunique())
plt.plot(nunique_aftertrim.index,nunique_aftertrim['ticker'],'r')
plt.legend('Number of firms with a median of daily tweets bigger than 50',loc='upper right')
plt.title('Event Study Coverage')
plt.show()


nunique_tweetDB = pd.DataFrame(df[['date','ticker']].groupby('date')['ticker'].nunique())
plt.plot(nunique_tweetDB.index,nunique_tweetDB['ticker'],'r')
plt.legend('Number of firms covered daily by at least one tweet',loc='upper right')
plt.title('Stocktwits Coverage')
plt.show()

#################################### unique messages ########################################################################

nb_unique_msg = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D', offset = timedelta(hours=-8))]).apply(lambda x: len(x.id_msg.unique())))
nb_unique_msg.rename(columns={0:'count'}, inplace=True)
nb_unique_msg.index = nb_unique_msg.index  + timedelta(days=1)
nb_unique_msg.index  = nb_unique_msg.index.normalize()
nb_unique_msg = pd.merge(nb_unique_msg, match_series_copy,  how='left', left_index=True, right_on = 'date')

nb_unique_msg['count_adj'] = 0

Nc = 0
for index, row in nb_unique_msg.iterrows():
    if row['businessday']==False:
        Nc += row['count']
    else:
        nb_unique_msg.loc[index,'count_adj'] = nb_unique_msg.loc[index,'count'] + Nc
        Nc = 0

nb_unique_msg = nb_unique_msg.drop(nb_unique_msg[nb_unique_msg['businessday']==False].index | nb_unique_msg[nb_unique_msg['date']<pd.to_datetime(datetime(2010,1,1), utc = True)].index).reset_index(drop=True)
nb_unique_msg['date'] = pd.to_datetime(nb_unique_msg['date'], utc = True)
nb_unique_msg = pd.merge(nb_unique_msg, sentratio_and_price.loc[sentratio_and_price['ticker'] == "AAPL",['date','total_vol_tweet']],  how='left', left_on='date', right_on = 'date')


plt.plot(nb_unique_msg['date'], nb_unique_msg['total_vol_tweet'],'r')
plt.plot(nb_unique_msg['date'], nb_unique_msg['count_adj'],'b')
plt.legend(('Number of messages (with double counting)', 'Number of messages (without double counting)'),loc='upper left')
plt.title('Total number of messages (daily agg.)')
plt.show()

from textwrap import wrap
plt.plot(nb_unique_msg['date'], nb_unique_msg['total_vol_tweet']/nb_unique_msg['count_adj'])
plt.title("\n".join(wrap('# messages with double counting divided by # messages without double counting', 45)))
plt.show()

#print histogram of number of messages per number of tickers tagged in the message
dff = pd.DataFrame(df[['ticker','id_msg']].groupby('id_msg').count()).groupby('ticker').size()
plt.bar(dff[1:].index,dff[1:])
plt.title('# of messages per # of tickers tagged in the message')
plt.xlabel('# of tickers in the message')
plt.ylabel('# of messages')
plt.show()

#################################### double events? ########################################################################

sentratio_and_price[(sentratio_and_price['Event3'] * sentratio_and_price['Event3'].shift(-1)) == 1].index


#################################### systematic shift in cumRet 1 day before event ########################################################################

print(np.mean(eventstudy.loc[(eventstudy['d']==-4) & (eventstudy['Type'] == "Good"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==-3) & (eventstudy['Type'] == "Good"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==-2) & (eventstudy['Type'] == "Good"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==-1) & (eventstudy['Type'] == "Good"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==0) & (eventstudy['Type'] == "Good"),'cumRet']))

print(np.mean(eventstudy.loc[(eventstudy['d']==-4) & (eventstudy['Type'] == "Bad"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==-3) & (eventstudy['Type'] == "Bad"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==-2) & (eventstudy['Type'] == "Bad"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==-1) & (eventstudy['Type'] == "Bad"),'cumRet']))
print(np.mean(eventstudy.loc[(eventstudy['d']==0) & (eventstudy['Type'] == "Bad"),'cumRet']))



test_distrib(eventstudy, -2, 'cumRet')

#################################### events followed by events #########################################################

Bull_Bull, Bull_Neut, Bull_Bear, Neut_Bull, Neut_Neut, Neut_Bear, Bear_Bull, Bear_Neut, Bear_Bear = 0, 0, 0, 0, 0, 0, 0, 0, 0
dic = {'Bull_Bull':0, 'Bull_Neut':0, 'Bull_Bear':0, 'Neut_Bull':0, 'Neut_Neut':0, 'Neut_Bear':0, 'Bear_Bull':0, 'Bear_Neut':0, 'Bear_Bear':0}

def cnt_follow(x):
        for i in x.index:
            if len(x.loc[i + 1:i + 1, 'Type']) > 0:
                if x.loc[i,'Type'] == 'Good':
                    if   x.loc[i+1:i+1,'Type'].values[0] == 'Good': dic['Bull_Bull'] =+1
                    elif x.loc[i+1:i+1,'Type'].values[0] == 'Neutral': dic['Bull_Neut'] =+1
                    elif x.loc[i+1:i+1, 'Type'].values[0] == 'Bad': dic['Bull_Bear'] = +1
                elif x.loc[i,'Type'] == 'Neutral':
                    if   x.loc[i+1:i+1,'Type'].values[0] == 'Good': dic['Neut_Bull'] =+1
                    elif x.loc[i+1:i+1,'Type'].values[0] == 'Neutral': dic['Neut_Neut'] =+1
                    elif x.loc[i+1:i+1,'Type'].values[0] == 'Bad': dic['Neut_Bear'] = +1
                elif x.loc[i,'Type'] == 'Bad':
                    if   x.loc[i+1:i+1,'Type'].values[0] == 'Good': dic['Bear_Bull'] =+1
                    elif x.loc[i+1:i+1,'Type'].values[0] == 'Neutral': dic['Bear_Neut'] =+1
                    elif x.loc[i+1:i+1, 'Type'].values[0] == 'Bad': dic['Bear_Bear'] = +1
        return dic

dic = eventlist.groupby('ticker').apply(cnt_follow)
            
Bull_Bull, Bull_Neut, Bull_Bear = Bull_Bull/np.sum(eventstudy['Type']=='Good'), \
                                  Bull_Neut/np.sum(eventstudy['Type']=='Good'), Bull_Bear /np.sum(eventstudy['Type']=='Good')

Neut_Bull, Neut_Neut, Neut_Bear = Neut_Bull/np.sum(eventstudy['Type']=='Good'), \
                                  Neut_Neut/np.sum(eventstudy['Type']=='Good'), Neut_Bear /np.sum(eventstudy['Type']=='Good')

Bear_Bull, Bear_Neut, Bear_Bear = Bear_Bull/np.sum(eventstudy['Type']=='Good'), \
                                  Bear_Neut/np.sum(eventstudy['Type']=='Good'), Bear_Bear /np.sum(eventstudy['Type']=='Good')

print("{:. 0%}".format(Bull_Bull))


#################################### portfolios ########################################################################

#with open('sentratio_and_price_for_pf2.pkl', 'wb' ) as f:     #sentratio_and_price_for_pf
#    pickle.dump(sentratio_and_price, f)


sentratio_and_price = pd.read_pickle('sentratio_and_price_for_pf2.pkl')

CAP_WINDOW = 14

sentratio_and_price['CAP'] =  sentratio_and_price.groupby(['ticker'])['abnormal_pol'].cumsum() - sentratio_and_price.groupby(['ticker'])['abnormal_pol'].cumsum().shift(CAP_WINDOW)

#reset CAP to 0 everytime there is an event
sentratio_and_price['CAPwithReset'] = sentratio_and_price['CAP']

def cumsumafterevent(x):
    for i in x.index:
        if x.loc[i,'Event3']==True:
            x.loc[i, 'CAPwithReset'] = 0
            x.loc[i+1:i+CAP_WINDOW, 'CAPwithReset'] = sentratio_and_price.loc[i+1:i+CAP_WINDOW,'abnormal_pol'].cumsum()
    return x

sentratio_and_price = sentratio_and_price.groupby('ticker').apply(cumsumafterevent)

sentratio_and_price.loc[(sentratio_and_price['date'] == pd.to_datetime(pd.datetime(2012, 3, 8), utc=True)) &
                        (sentratio_and_price['ticker'] == 'UVXY'), 'daily_return'] = -0.08

threshold_bigfirm = 50

N_byfirm['big'] = N_byfirm['N'] > threshold_bigfirm
tickers_to_keep = list(N_byfirm[N_byfirm['big']].index)
print('there are currently ' + str(tickers_to_keep.__len__()) + ' firms in the sample')

sentratio_and_price = sentratio_and_price[sentratio_and_price['ticker'].isin(tickers_to_keep)]
sentratio_and_price = sentratio_and_price.reset_index(drop=True)

sentratio_and_price['CAP'] = sentratio_and_price['CAPwithReset']

CAP = pd.pivot_table(sentratio_and_price, values = 'CAP', index = 'date', columns='ticker',dropna = False)
CAP = CAP[CAP_WINDOW:]
CAP = CAP.replace(0, np.nan)

RET = pd.pivot_table(sentratio_and_price, values = 'daily_return', index = 'date', columns='ticker',dropna = False)
RET = RET[CAP_WINDOW:]
RET = RET.replace(np.nan, 0)


##########################################

PfWeights = RET.copy()
PfWeights.loc[:, :] = 0

sentratio_and_price = sentratio_and_price[sentratio_and_price['date'] > pd.to_datetime(pd.datetime(2011, 1, 28), utc=True)]
sentratio_and_price['next5'] = np.nan

horizon = 2
def next5(x,horizon):
    for i in x.index:
        if any(x.loc[i+1:i+horizon-1,'Type']=='Neutral') == True:
            x.loc[i,'next5'] = 'Neutral'
        if any(x.loc[i+1:i+horizon-1,'Type']=='Bad') == True:
            x.loc[i,'next5'] = 'Bad'
        if any(x.loc[i+1:i+horizon-1,'Type']=='Good') == True:
            x.loc[i,'next5'] = 'Good'
        if all(x.loc[i+1:i + horizon-1, 'Type'].isna()) == True:
            x.loc[i, 'next5'] = 'None'
    return x

sentratio_and_price = sentratio_and_price.groupby('ticker').apply(next5, horizon=horizon)

rf = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\tweet_sklearn\\rf.csv",sep=';')
rf['date'] = pd.to_datetime(rf['caldt'],format="%Y%m%d")
rf['date'] = pd.to_datetime(rf['date'], utc = True)

sentratio_and_price = pd.merge(sentratio_and_price, rf[['date','t90ret']],  how='left', left_on='date', right_on = 'date')
sentratio_and_price['t90ret'] = sentratio_and_price['t90ret'].ffill()
sentratio_and_price['t90ret'] = sentratio_and_price['t90ret'].fillna(0)
sentratio_and_price['excess_return'] = sentratio_and_price['daily_return'] - sentratio_and_price['t90ret']


def perf(x):
    return np.log(1+x).sum()

## daily distributions of CAP



def create_portfolios(data, sigmas, strat = 'long', plots = 'show'):
    sentratio_and_price = data.copy()

    if plots == 'create':
        folder = "sigma " + str("{0:.2f}".format(sigmas))
        path = os.path.join(os.getcwd(), 'results', 'Portfolios','Dynamic Thresholds' , 'CAP14', folder)
        if not os.path.exists(path):
            os.makedirs(path)

    plt.plot(sentratio_and_price[['date','CAP']].groupby('date').mean(),'b')
    plt.plot(sentratio_and_price[['date','CAP']].groupby('date').mean()+sigmas*sentratio_and_price[['date','CAP']].groupby('date').std(),'g')
    plt.plot(sentratio_and_price[['date','CAP']].groupby('date').mean()-sigmas*sentratio_and_price[['date','CAP']].groupby('date').std(),'r')
    plt.legend(('Mean','Mean+' + str(sigmas) + 'std', 'Mean-' + str(sigmas) + 'std'))
    plt.title('Cross-sectional statistics of CAP')
    plt.xlabel('Date')
    if plots =='create':
        plt.savefig(os.path.join(path, 'cross_sectional_cap.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()


    mean = sentratio_and_price[['date','CAP']].groupby('date').mean().rename(columns={'CAP':'CAPmean'})
    up = sentratio_and_price[['date','CAP']].groupby('date').mean() + sigmas*sentratio_and_price[['date','CAP']].groupby('date').std()
    up = up.rename(columns={'CAP':'CAP_up'})
    down = sentratio_and_price[['date','CAP']].groupby('date').mean() - sigmas*sentratio_and_price[['date','CAP']].groupby('date').std()
    down = down.rename(columns={'CAP':'CAP_down'})

    sentratio_and_price['CAPmean'] = pd.merge(sentratio_and_price[['date']],mean,left_on='date',right_index=True)['CAPmean']
    sentratio_and_price['CAP_up'] = pd.merge(sentratio_and_price[['date']],up,left_on='date',right_index=True)['CAP_up']
    sentratio_and_price['CAP_down'] = pd.merge(sentratio_and_price[['date']],down,left_on='date',right_index=True)['CAP_down']

    sentratio_and_price['long_position'] = 1*(sentratio_and_price['CAP'] > sentratio_and_price['CAP_up'])
    sentratio_and_price['short_position'] = 1*(sentratio_and_price['CAP'] < sentratio_and_price['CAP_down'])

    sentratio_and_price.loc[sentratio_and_price['excess_return'].isna(), 'opti_position'] = 0
    sentratio_and_price.loc[sentratio_and_price['excess_return'].isna(), 'long_position'] = 0
    sentratio_and_price.loc[sentratio_and_price['excess_return'].isna(), 'short_position'] = 0

    LONGPOS = pd.pivot_table(sentratio_and_price, values = 'long_position', index = 'date', columns='ticker',dropna = False)
    SHORTPOS = pd.pivot_table(sentratio_and_price, values = 'short_position', index = 'date', columns='ticker',dropna = False)

    EXCESSRET = pd.pivot_table(sentratio_and_price, values = 'excess_return', index = 'date', columns='ticker',dropna = False)
    EXCESSRET = EXCESSRET.replace(np.nan, 0)

    holding_period = 1
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=holding_period)
    EXCESSRET_next5 =(1.+ EXCESSRET).shift(-1).rolling(window=indexer).agg(lambda x : x.prod()) -1

    PF_LONG = np.sum(LONGPOS * EXCESSRET_next5 , axis = 1) / np.sum(LONGPOS!=0, axis = 1)
    PF_SHORT = np.sum(SHORTPOS * EXCESSRET_next5 , axis = 1) / np.sum(SHORTPOS!=0, axis = 1)
    PFS = pd.merge(pd.DataFrame({'Bullish': PF_LONG}), pd.DataFrame({'Bearish': PF_SHORT}), how='left', right_index=True,
                   left_index=True)

    plt.plot(PF_LONG.index, PF_LONG, color='green', label='Bullish')
    plt.plot(PF_SHORT.index, PF_SHORT, color='red', label='Bearish')
    plt.legend()
    plt.title(str(holding_period) + '-days portfolio returns')
    plt.xlabel('Date')
    plt.ylabel('Portfolio return')
    if plots =='create':
        plt.savefig(os.path.join(path, 'returns.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()

    ax = sns.boxplot(data=PFS, whis=1.5, palette={"Bullish": "green", "Bearish": "red"})
    plt.title(str(holding_period) + '-days portfolio returns')
    if plots =='create':
        plt.savefig(os.path.join(path, 'boxplot.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()

    ax = sns.boxplot(data=PFS, whis=1.5, palette={"Bullish": "green", "Bearish": "red"})
    plt.title(str(holding_period) + '-days portfolio returns')
    plt.ylim([-0.1, 0.1])
    if plots =='create':
        plt.savefig(os.path.join(path, 'boxplot_zoomed.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()


    keep = np.sum(RET, axis=1) != 0
    plt.plot(LONGPOS.loc[keep, :].index, np.sum(LONGPOS.loc[keep, :] != 0, axis=1), color='green', label='Bullish')
    plt.plot(SHORTPOS.loc[keep, :].index, np.sum(SHORTPOS.loc[keep, :] != 0, axis=1), color='red',
             label='Bearish')
    plt.legend()
    plt.title('Number of positions through the years')
    plt.xlabel('Date')
    plt.ylabel('Number of positions')
    if plots =='create':
        plt.savefig(os.path.join(path, 'positions.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()


    labels, counts = np.unique(np.sum(LONGPOS != 0, axis=1), return_counts=True)
    plt.bar(labels, counts, width=0.5, color = 'C0')
    plt.title('Number of positions per bullish portfolio')
    plt.xlabel('Number of positions')
    plt.ylabel('Number of portfolios')
    plt.xticks(ticks=list(range(np.max(np.sum(LONGPOS != 0, axis=1)) + 1)),
               labels=list(range(np.max(np.sum(LONGPOS != 0, axis=1)) + 1)))
    if plots == 'create':
        plt.savefig(os.path.join(path, 'distribution_long_positions.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()

    labels, counts = np.unique(np.sum(SHORTPOS != 0, axis=1), return_counts=True)
    plt.bar(labels, counts, width=0.5, color = 'C0')
    plt.xticks(ticks=list(range(np.max(np.sum(SHORTPOS != 0, axis=1))+1)),labels=list(range(np.max(np.sum(SHORTPOS != 0, axis=1))+1)))
    plt.title('Number of positions per bearish portfolio')
    plt.xlabel('Number of positions')
    plt.ylabel('Number of portfolios')
    if plots == 'create':
        plt.savefig(os.path.join(path, 'distribution_short_positions.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()

    labels_bull, counts_bull = np.unique(np.sum(LONGPOS != 0, axis=1), return_counts=True)
    labels_bear, counts_bear = np.unique(np.sum(SHORTPOS != 0, axis=1), return_counts=True)
    plt.bar(labels_bull-0.2, counts_bull, width=0.4, color = 'green', label = 'Bullish')
    plt.bar(labels_bear+0.2, counts_bear, width=0.4, color='red', label='Bearish')
    plt.xticks(ticks=list(range(np.max(labels_bull.tolist() + labels_bear.tolist())+1)),labels=list(range(np.max(labels_bull.tolist() + labels_bear.tolist())+1)))
    plt.title('Number of positions per portfolio')
    plt.xlabel('Number of positions')
    plt.ylabel('Number of portfolios')
    plt.legend()
    if plots == 'create':
        plt.savefig(os.path.join(path, 'distribution_positions.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()


    # daily portfolio  #-3   5  looks ok
    plt.plot(PFS.index, PFS.loc[:, 'Bullish'].apply(perf).fillna(0).cumsum(), 'g')
    plt.plot(PFS.index, PFS.loc[:, 'Bearish'].apply(perf).fillna(0).cumsum(), 'r')
    plt.plot(sentratio_and_price.loc[sentratio_and_price['ticker'] == 'AAPL', 'date'],
             sentratio_and_price.loc[sentratio_and_price['ticker'] == 'AAPL', 'market_return'].apply(perf).fillna(
                 0).cumsum(), 'b')
    plt.legend(('Bullish Portfolio', 'Bearish Portfolio', 'SPY'))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    if plots =='create':
        plt.savefig(os.path.join(path, 'cumulative_log_return.jpg'))
        plt.close()
    elif plots == 'show':
        plt.show()
    else:
        plt.close()

    # conf matrix

    def optipos(x):
        if x['long_position'] == 1:
            return 1
        elif x['short_position'] == 1:
            return -1
        else:
            return 0

    sentratio_and_price['opti_position'] = sentratio_and_price.apply(lambda x : optipos(x), axis = 1)

    Bull_Long = np.sum((sentratio_and_price['next5'] == 'Good') & (sentratio_and_price['opti_position'] == 1))
    Bull_Short = np.sum((sentratio_and_price['next5']== 'Good') & (sentratio_and_price['opti_position'] == -1))
    Bull_None = np.sum((sentratio_and_price['next5'] == 'Good') & (sentratio_and_price['opti_position'] == 0))
    Bear_Long = np.sum((sentratio_and_price['next5'] == 'Bad') & (sentratio_and_price['opti_position'] == 1))
    Bear_Short = np.sum((sentratio_and_price['next5']== 'Bad') & (sentratio_and_price['opti_position'] == -1))
    Bear_None = np.sum((sentratio_and_price['next5']== 'Bad') & (sentratio_and_price['opti_position'] == 0))
    Neut_Long = np.sum((sentratio_and_price['next5'] == 'Neutral') & (sentratio_and_price['opti_position'] == 1))
    Neut_Short = np.sum((sentratio_and_price['next5']  == 'Neutral') & (sentratio_and_price['opti_position'] == -1))
    Neut_None = np.sum((sentratio_and_price['next5']  == 'Neutral') & (sentratio_and_price['opti_position'] == 0))
    NoEvent_Long = np.sum((sentratio_and_price['next5']  == 'None') & (sentratio_and_price['opti_position'] == 1))
    NoEvent_Short = np.sum((sentratio_and_price['next5']  == 'None') & (sentratio_and_price['opti_position'] == -1))
    NoEvent_None = np.sum((sentratio_and_price['next5']  == 'None') & (sentratio_and_price['opti_position'] == 0))

    measures = dict()
    if strat == "long":
        measures['TP'] = Bull_Long
        measures['FN'] = Bull_Short
        measures['FP'] = Bear_Long + Neut_Long + NoEvent_Long
        measures['TN'] = Bear_Short + Neut_Short + NoEvent_Short

    if strat == "short":
        measures['TP'] = Bear_Short
        measures['FN'] = Bear_Long
        measures['FP'] = Bull_Short + Neut_Short + NoEvent_Short
        measures['TN'] = Bull_Long + Neut_Long + NoEvent_Long

    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FN'].sum().sum())
    measures['TPR'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FN'].sum().sum())
    measures['FPR'] = measures['FP'].sum().sum() / (measures['FP'].sum().sum() + measures['TN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    measures['f1score'] = 2 * measures['precision'] * measures['recall'] / (measures['precision'] + measures['recall'])

    print('-' * 15, 'sigmas :', sigmas, '-' * 15)
    print('Bullish events, long positioned : ', Bull_Long)
    print('Bullish events, short positioned : ', Bull_Short)
    print('Bullish events, no position : ', Bull_None)
    print('Bearish events, long positioned : ', Bear_Long)
    print('Bearish events, short position : ', Bear_Short)
    print('Bearish events, no position : ', Bear_None)
    print('Neutral events, long positioned : ', Neut_Long)
    print('Neutral events, short position : ', Neut_Short)
    print('Neutral events, no position : ', Neut_None)
    print('No events, no position : ', NoEvent_None)
    print('No events, long position : ', NoEvent_Long)
    print('No events, short position : ', NoEvent_Short)
    print(measures)
    return measures, sentratio_and_price
#
# span = np.linspace(-3,3,20)
# obs1  = []
# for sig in span:
#     measures,_ = create_portfolios(sentratio_and_price, sig, 'long', plots='no')
#     obs1.append([sig, measures['f1score'], measures['precision'], measures['recall'], measures['TPR'], measures['FPR']])
#
# obs2  = []
# for sig in span:
#     measures,_ = create_portfolios(sentratio_and_price, sig, 'short', plots='no')
#     obs2.append([sig, measures['f1score'], measures['precision'], measures['recall'], measures['TPR'], measures['FPR']])
#
# f1_long = [x[1] for x in obs1]
# f1_short = [x[1] for x in obs2]
# max_long = obs1[np.nanargmax(f1_long)]
# max_short = obs2[np.nanargmax(f1_short)]
#
# plt.style.use('seaborn-whitegrid')
# plt.plot([item[0] for item in obs1], [item[1] for item in obs1], '-', color='green',label ='Long')
# plt.plot([item[0] for item in obs2], [item[1] for item in obs2], '-', color='red', label='Short')
# plt.scatter(max_short[0], max_short[1], s=80, facecolors='none', edgecolors='blue')
# plt.scatter(max_long[0], max_long[1], s=80, facecolors='none', edgecolors='blue')
# #plt.axvline(x=max2[0],  ymax=0.931, color = 'blue',  linestyle='--')
# #plt.axvline(x=max1[0], ymax= 0.921, color = 'blue',  linestyle='--')
# plt.legend()
# plt.xlabel('Threshold')
# plt.ylabel('F1 score')
# plt.title('CAP thresholds maximizing F1 score')
# #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
# plt.show()
#
# plt.style.use('seaborn-whitegrid')
# plt.plot([item[3] for item in obs1], [item[2] for item in obs1], '-', color='green',label ='Long')
# plt.plot([item[3] for item in obs2], [item[2] for item in obs2], '-', color='red', label='Short')
# #plt.axvline(x=max2[0],  ymax=0.931, color = 'blue',  linestyle='--')
# #plt.axvline(x=max1[0], ymax= 0.921, color = 'blue',  linestyle='--')
# plt.legend()
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-recall curve')
# #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
# plt.show()
#
# from numpy import trapz
# area_long = trapz(y=[item[4] for item in obs1], x=[item[5] for item in obs1])
# print("AUC Long =", -area_long)
# area_short = trapz(y=[item[4] for item in obs2], x=[item[5] for item in obs2])
# print("AUC Short =", area_short)
#
#
# plt.style.use('seaborn-whitegrid')
# plt.plot([item[5] for item in obs1], [item[4] for item in obs1], '-', color='green',label ='Long, AUC = ' + str("{:.2f}".format(-area_long)))
# plt.plot([item[5] for item in obs2], [item[4] for item in obs2], '-', color='red', label='Short, AUC = ' + str("{:.2f}".format(area_short)))
# #plt.axvline(x=max2[0],  ymax=0.931, color = 'blue',  linestyle='--')
# #plt.axvline(x=max1[0], ymax= 0.921, color = 'blue',  linestyle='--')
# plt.legend()
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC curve')
# #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
# plt.show()
#
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx
#
# prec_long = [x[2] for x in obs1]
# max_prec_long = obs1[np.nanargmax(prec_long)]
# max_prec_short = obs2[find_nearest([x[3] for x in obs2], 0.17)]
#
#
# plt.style.use('seaborn-whitegrid')
# plt.plot([item[0] for item in obs1], [item[1] for item in obs1], '-', color='green',label ='Long')
# plt.plot([item[0] for item in obs2], [item[1] for item in obs2], '-', color='red', label='Short')
# plt.scatter(max_prec_short[0], max_prec_short[1], s=80, facecolors='none', edgecolors='blue')
# plt.scatter(max_prec_long[0], max_prec_long[1], s=80, facecolors='none', edgecolors='blue')
# #plt.axvline(x=max2[0],  ymax=0.931, color = 'blue',  linestyle='--')
# #plt.axvline(x=max1[0], ymax= 0.921, color = 'blue',  linestyle='--')
# plt.legend()
# plt.xlabel('Threshold')
# plt.ylabel('F1 score')
# plt.title('CAP thresholds maximizing precision')
# #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
# plt.show()


create_portfolios(sentratio_and_price, 2.58, 'long', plots='create')
create_portfolios(sentratio_and_price, 1.96, 'long', plots='create')
create_portfolios(sentratio_and_price, 2.81, 'long', plots='create')

