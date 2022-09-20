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

#df = pd.read_pickle('df_clean.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_1.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_2.pkl')
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_new.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_new.pkl')])

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

#sentratio_and_price = pd.read_pickle('sentratio_and_price.pkl')

#=========================== volume of daily tweets across time ============================

daily_volume_total = pd.DataFrame(df[['date','sent_merged']].set_index('date').groupby(pd.Grouper(freq='D')).count())
weekly_volume_total = pd.DataFrame(df[['date','sent_merged']].set_index('date').groupby(pd.Grouper(freq='W')).count())

plt.plot(daily_volume_total.index,daily_volume_total.sent_merged)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Tweets volume (daily agg.)')
plt.show()

plt.plot(weekly_volume_total.index[:-1],weekly_volume_total.sent_merged[:-1])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Tweets volume (weekly agg.)')
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
from itertools import product

tickers = df['ticker'].unique().tolist()
uniques = [pd.date_range(start='2012-01-01', end='2020-03-23', freq='D').tolist(), tickers]   #todo start='2012-01-01', end='2020-03-23'
cadre = pd.DataFrame(product(*uniques), columns = ['date','ticker'])
cadre = cadre.sort_values(['ticker', 'date'], ascending=[1, 1])

#=========================== populate ================================================

sentratio = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D'),'ticker']).apply(Nbullish))
sentratio.rename(columns={0:'Nbullish'}, inplace=True)
sentratio['Nbearish'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D'),'ticker']).apply(Nbearish))
sentratio['N'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='D'),'ticker'])['sent_merged'].count())['sent_merged']
sentratio['Polarity'] = (sentratio['Nbullish']-sentratio['Nbearish'])/(10+sentratio['Nbearish']+sentratio['Nbullish']) # I add 10 to the denom so polarity on days with few bullish and 0 bearish are close to 0
sentratio['Agreement'] = 1-np.sqrt(1-((sentratio['Nbullish']-sentratio['Nbearish'])/(10+sentratio['Nbearish']+sentratio['Nbullish']))**2)
sentratio = sentratio.reset_index()

sentratio = pd.merge(cadre, sentratio,  how='left', left_on=['date','ticker'], right_on = ['date','ticker'])
sentratio = sentratio.fillna(0)

#============================== save =========================================================#

#with open('dailypolarity2.pkl', 'wb' ) as f:
#    pickle.dump(sentratio, f)

#============================== load =========================================================#

sentratio = pd.read_pickle('dailypolarity2.pkl')

N_byfirm = pd.DataFrame(sentratio[sentratio['date']>'2013-01-01'].groupby('ticker')['N'].median())

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

threshold_bigfirm = 50
N_byfirm['big'] = N_byfirm['N'] > threshold_bigfirm
tickers_to_keep = list(N_byfirm[N_byfirm['big']].index)


sentratio = sentratio[sentratio['ticker'].isin(tickers_to_keep)]
sentratio = sentratio.reset_index(drop=True)


#============================== merge sentiment and market var =========================================================#

matching_table = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'table_de_corr_changetickers.csv'),sep=';')

prices_db = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\tweet_sklearn\\daily_crsp.csv",sep=',')
prices_db.TICKER = prices_db.TICKER.replace(matching_table.set_index('old').new.to_dict()) #correct tickers with matching table
prices_db['date'] = pd.to_datetime(prices_db['date'],format="%Y%m%d")
prices_db['adjprice'] = prices_db['PRC']/prices_db['CFACPR']
prices_db = prices_db.sort_values(['TICKER', 'date'], ascending=[1, 1])
prices_db['daily_return'] = prices_db.groupby('TICKER').adjprice.pct_change()

sentratio_and_price = pd.merge(sentratio, prices_db,  how='left', left_on=['date','ticker'], right_on = ['date','TICKER'])

sentratio_and_price = sentratio_and_price.drop(['PERMNO','COMNAM', 'PERMCO','ISSUNO','CUSIP','BIDLO','ASKHI','PRC','RET','BID','ASK',
                                                'CFACPR','CFACSHR','RET','CFACPR','CFACSHR', 'TICKER','OPENPRC','NUMTRD','adjprice'], axis=1)

#with open('sentratio_and_price.pkl', 'wb' ) as f:
#    pickle.dump(sentratio_and_price, f)
#===============================  activity inside/outside trading days =====================================
# users post three times more during trading days than outside trading days
outside = sentratio_and_price[sentratio_and_price['daily_return'].isna()]['N'].mean()
inside = sentratio_and_price[~sentratio_and_price['daily_return'].isna()]['N'].mean()
print('avg posts inside/outside trading days : ', int(inside) ,'/', int(outside))

#===============================  define events =====================================

sentratio_and_price['medianN'] = sentratio_and_price.groupby('ticker')['N'].rolling(31, center = True).median().reset_index(0,drop=True)
sentratio_and_price['phi'] = (sentratio_and_price['N'] - sentratio_and_price['medianN']) / np.maximum(sentratio_and_price['medianN'],10)
sentratio_and_price['Event'] = sentratio_and_price['phi']  > 4
sentratio_and_price.loc[sentratio_and_price['date'] < '2013-02-01', ['Event']] = False

#===============================  create eventlist =====================================

sentratio_and_price['d'] = np.nan

events_index = sentratio_and_price[sentratio_and_price['Event']].index
eventlist = []
for e in events_index:
    eventlist.append([sentratio_and_price.loc[e,'date'],sentratio_and_price.loc[e,'ticker'], e])
eventlist = pd.DataFrame(eventlist, columns = ['date','ticker','Index'])

#===============================  standardize polarity =====================================

spy = sentratio_and_price[sentratio_and_price['ticker']=='SPY']
sentratio_and_price = pd.merge(sentratio_and_price, spy[['date','daily_return']],  how='left', left_on='date', right_on = 'date')
sentratio_and_price.rename(columns={'daily_return_x':'daily_return','daily_return_y':'market_return'}, inplace=True)

market_polarity = pd.read_pickle('market_polarity.pkl')
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

sentratio_and_price['mm_params_ret'] = sentratio_and_price.loc[sentratio_and_price['date']>'2013-01-01',:].apply(lambda row: mm_ret(row['ticker'], row['date']), axis=1)

sentratio_and_price['mm_params_pol'] = sentratio_and_price.loc[sentratio_and_price['date']>'2013-01-01',:].apply(lambda row: mm_pol(row['ticker'], row['date']), axis=1)
sentratio_and_price['cm_normal_pol'] = sentratio_and_price.loc[sentratio_and_price['date']>'2013-01-01',:].apply(lambda row: cm_pol(row['ticker'], row['date']), axis=1)

sentratio_and_price = sentratio_and_price[sentratio_and_price['date'] > '2013-01-01']

sentratio_and_price['normal_return'] = sentratio_and_price['mm_params_ret'].map(lambda x: x[0]) + sentratio_and_price['market_return'] * sentratio_and_price.mm_params_ret.map(lambda x: x[1])
sentratio_and_price['mm_normal_pol'] = sentratio_and_price['mm_params_pol'].map(lambda x: x[0]) + sentratio_and_price['market_polarity'] * sentratio_and_price.mm_params_pol.map(lambda x: x[1])

sentratio_and_price['abnormal_return'] = sentratio_and_price['daily_return'] - sentratio_and_price['normal_return']
#sentratio_and_price['abnormal_pol'] = sentratio_and_price['Polarity'] - sentratio_and_price['cm_normal_pol']   #chose this for constant mean normal polarity
sentratio_and_price['abnormal_pol'] = sentratio_and_price['Polarity'] - sentratio_and_price['mm_normal_pol']  #chose this for market model normal polarity



#===============================  distribution polarities of events =====================================
from sklearn.neighbors import KernelDensity

abn_pol_events = sentratio_and_price.loc[(sentratio_and_price['Event'] == True),'abnormal_pol']
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

abn_pol_all = sentratio_and_price['abnormal_pol']
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

low_thresh = np.quantile(abn_pol_events,1/3)
up_thresh = np.quantile(abn_pol_events,2/3)

low_thresh = -0.04
up_thresh = 0.08

#===============================  define type of events =====================================

sentratio_and_price['Type'] = np.nan
sentratio_and_price.loc[(sentratio_and_price['Event'] == True) & (sentratio_and_price['abnormal_pol']>up_thresh),'Type'] = 'Good'
sentratio_and_price.loc[(sentratio_and_price['Event'] == True) & (sentratio_and_price['abnormal_pol']<low_thresh),'Type'] = 'Bad'
sentratio_and_price.loc[(sentratio_and_price['Event'] == True) & (((sentratio_and_price['abnormal_pol'])<up_thresh) & ((sentratio_and_price['abnormal_pol'])>low_thresh)),'Type'] = 'Neutral'

print('# good/neutral/bad events: ' , np.sum(sentratio_and_price['Type']=='Good'),
      ' / ', np.sum(sentratio_and_price['Type']=='Neutral'), ' / ', np.sum(sentratio_and_price['Type']=='Bad'))
# good/neutral/bad events:  240  /  232  /  233

eventlist = pd.merge(eventlist[['date','ticker','Index']], sentratio_and_price[['date','ticker','Type']],  how='left', left_on=['date','ticker'], right_on = ['date','ticker'])



#===============================  event study =====================================


eventstudy = pd.DataFrame()
for _, row in eventlist.iterrows():
    cut_tick = sentratio_and_price[sentratio_and_price['ticker'] == row['ticker']].copy()
    cut_window = cut_tick.loc[row['Index']-20:row['Index']+20].copy()
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
plt.xlabel('days')
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
plt.xlabel('days')
plt.ylabel('CAAP')
plt.grid(b=None)
plt.tick_params(axis='both', labelsize=10, length = 3)
plt.legend(('Bullish', 'Neutral','Bearish'),
           loc='upper left')
#plt.savefig('ES_CAAR.jpg')
plt.show()

#todo very ncie plots. it looks like polarity trend is more persistent. people are biased towards the past instead of looking for the future.
#todo return: consistent with previous literature and EMH : adjust quickly

#===============================  plots =====================================


eventlist = pd.merge(eventlist[['date','ticker','Index','Type']], sentratio_and_price[['date', 'ticker', 'N',]], how='left', left_on=['ticker', 'date'],
                     right_on=['ticker', 'date'])

def plot_activity(ticker):
    cut_act = sentratio_and_price[sentratio_and_price['ticker']==ticker]
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

plot_activity('FB')

#=============================== events across time ================================

types = ['Bad','Neutral','Good']
uniques = [pd.date_range(start='2013-02-11', end='2020-02-26', freq='M').tolist(), types]
cadre = pd.DataFrame(product(*uniques), columns = ['date','Type'])
cadre.rename(columns={0:'date'}, inplace=True)
cadre = cadre.sort_values(['Type', 'date'], ascending=[1, 1])

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

def regress(subset,Xvar,Yvar):
    if(subset==None):
        sentratio_and_price_reg = sentratio_and_price.dropna(subset=['daily_return']).copy().reset_index(drop=True)
    else:
        sentratio_and_price_reg = sentratio_and_price[sentratio_and_price['ticker'].isin(subset)].dropna(subset=['daily_return']).copy().reset_index(drop=True)
    sentratio_and_price_reg = sentratio_and_price_reg[sentratio_and_price_reg['daily_return'].between(sentratio_and_price_reg['daily_return'].quantile(.05),
                                                                 sentratio_and_price_reg['daily_return'].quantile(.95))]  # without outliers
    sentratio_and_price_reg['daily_return_t+1'] = sentratio_and_price_reg.groupby(['ticker'])['daily_return'].shift(-1)
    sentratio_and_price_reg['abn_return_t+1'] = sentratio_and_price_reg.groupby(['ticker'])['abnormal_return'].shift(-1)
    sentratio_and_price_reg = sentratio_and_price_reg.dropna(subset=['daily_return_t+1','abn_return_t+1','abnormal_pol']).reset_index(drop=True)

    X = sm.add_constant(sentratio_and_price_reg[Xvar])  # --- with constant
    y = sentratio_and_price_reg[Yvar]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1)
    model = OLS(yTrain, xTrain)
    res = model.fit()
    print(res.summary())

regress(None,['abnormal_pol'],'abnormal_return')   #polarity signif
regress(None,['abnormal_pol'],'abn_return_t+1')    #polarity non signif

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
    n1 = len(dat[dat['Type'] == type[0]][var])
    n2 = len(dat[dat['Type'] == type[1]][var])
    Z = (U - n1*n2/2)/np.sqrt(n1*n2*(n1+n2+1)/12)

    if Z<-1.36:
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
MannWhitneyTest(eventstudy, -5, ['Good','Bad'], 'CAR')
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
plot_bullishness_return_company('TWTR')


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
plt.title('Tweets classification across time (month agg.)')
plt.show()


plt.stackplot(ts_neut.index,ts_neut['N_lab_bear']/ts_neut['N1']*100,
              (ts_neut['N_bear_class']-ts_neut['N_lab_bear'])/ts_neut['N1']*100,
              (ts_neut['N_neut_class'])/ts_neut['N1']*100,
              (ts_neut['N_bull_class']-ts_neut['N_lab_bull'])/ts_neut['N1']*100,
              (ts_neut['N_lab_bull'])/ts_neut['N1']*100,
             labels=['% labeled as bearish', '% classified as bearish','% classified as neutral','% classified as bullish', '% labeled as bullish'], colors = [(1, 0, 0, 1),(1, 0, 0, 0.4),'lightgray',(0.0, 0.5019607843137255, 0.0,0.4),(0.0, 0.5019607843137255, 0.0,1)])
plt.legend(reversed(plt.legend().legendHandles), reversed(['% labeled as bearish', '% classified as bearish','% classified as neutral','% classified as bullish', '% labeled as bullish']), frameon=True, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
plt.title('Tweets classification across time (month agg.)')
plt.show()



plt.stackplot(ts_neut.index,ts_neut['N_lab_bear']/ts_neut['N1']*100,  ts_neut['N_unlabeled']/ts_neut['N1']*100, ts_neut['N_lab_bull']/ts_neut['N1']*100
              , labels=['% labeled as bearish', '% unlabeled','% labeled as bullish'], colors = ['red','lightgray','green'])
plt.legend(reversed(plt.legend().legendHandles), reversed(['% labeled as bearish', '% unlabeled','% labeled as bullish']), frameon=True, loc = 'center left')
plt.title('User-labeled tweets across time (month agg.)')
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


import pdb

def xsquared(x):
    res = x**2
    test = 3
    pdb.set_trace()
    plop = 4
    return x

xsquared(3)

############

import numpy as np
from numpy import log,exp
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize
from math import ceil
#import PyQt5


# Black-scholes for put and call options
def BS(S, k, t, T, sigma, r=0, q=0, delta=1):
    tau = T - t
    stau = np.sqrt(tau)
    sigma2 = sigma ** 2

    d1 = (np.log(S / k) + (r - q + 0.5 * sigma2) * tau * delta) / (sigma * stau * np.sqrt(delta))
    d2 = d1 - sigma * stau * np.sqrt(delta)

    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)

    N3 = norm.cdf(-d2)
    N4 = norm.cdf(-d1)

    c = S * np.exp(-q * tau * delta) * N1 - np.exp(-r * tau * delta) * k * N2  # Call price
    p = k * np.exp(-r * tau * delta) * N3 - S * np.exp(-q * tau * delta) * N4  # Put price

    return c, p

data = pd.read_csv("C:\\Users\\divernoi\\Desktop\\SX5E_Impliedvols.csv",sep=";")
#data= data.drop(columns = "Unnamed: 0")
strikes = data["K\T"]
implied_vols = data.iloc[:,1:]

#Initializing parameters
r, q = 0,0
T1, T2 = 1, 1.5
T = [0,0.025,0.101,0.197,0.274,0.523,0.772,1.769]
t = 0
maturities = np.arange(0.4,2.01,0.01)
#The Closing price of SX5E in March 1st,2010 was 2772.70 as stated in the paper
S0 =  2772.70
K  = S0*strikes

# Computing the observed call prices
Call_obs = pd.DataFrame(0, index=range(161), columns=T)
Call_obs.iloc[:, 0] = np.array(np.maximum(S0 - K, np.zeros(161)))

nonzero_vol = implied_vols > 0

for i in range(len(K)):
    for j in range(len(T) - 1):
        if implied_vols.iloc[i, j] > 0:
            Call_obs.iloc[i, j + 1] = BS(S0, K[i], t, T[j + 1], implied_vols.iloc[i, j], r, q)[0]


# The Andreasen-Huge algorithm to get Call_model

def AH(nonzero_vol, dt, dK, x, i, Call_obs, K):
    m = (np.array(nonzero_vol.iloc[:-1, 1]) + np.array(nonzero_vol.iloc[1:, 1])) / 2
    M = np.array([ceil(i) for i in m])
    Nk = len(K)
    vol = np.zeros((Nk, 1))
    k = 1
    for nn in M:
        while k < M[nn]:
            vol[k] = x[nn]
            k += 1

        vol[k:] = x[-1]

    # Now finding the Matrix A
    z = 0.5 * dt / dK ** 2 * np.power(vol[1:], 2)
    z = z[:-1]
    D = np.tile(z, [1, 3])
    D[:, 0] = -D[:, 0]
    D[:, 1] = D[:, 1] * 2
    D[:, 2] = -D[:, 2]
    # First Line of matrix A
    A = np.zeros((1, Nk))
    # From Line 2 till line Nk - 2
    for j in range(len(K) - 2):
        C = np.concatenate((np.zeros((1, j)), D[j, :]), axis=None)
        C = np.concatenate((C, np.zeros((1, Nk - 3 - j))), axis=None).reshape(1, 161)
        A = np.vstack([A, C])

    # Last Line
    A = np.vstack([A, np.zeros((1, Nk))])
    A = A + np.eye(Nk)
    C = np.linalg.inv(A) @ Call_obs.iloc[:, i]

    return C


dK = K[1] - K[0] #The step is constant

for i in range(len(T) - 1):
    dt = T[i+1] - T[i]
    f  = lambda x : np.sum(np.multiply(nonzero_vol.iloc[:,i],
                                      np.power(AH(nonzero_vol,dt,dK,x,i,Call_obs,K) - Call_obs.iloc[:,i+1],
                                                  2)))
    #Initializing x_0
    x00 = np.multiply(S0*0.5*np.ones((len(K),1)),nonzero_vol.iloc[:,i].to_numpy().reshape(161,1))
    x00 = x00[x00!=0]
    xx = scipy.optimize.fmin(f,  x0 = x00, disp = False )
    C_new = AH(nonzero_vol,dt,dK,xx,i,Call_obs,K)
    Call_obs.iloc[:,i+1] = C_new

iv = np.zeros(np.shape(implied_vols))

for i in range(len(K)):
    for j in range(1, len(T)):
        c = Call_obs.iloc[i, j]
        g = lambda sig: (BS(S0, K[i], t, T[j], sig)[0] - c) ** 2
        iv[i, j - 1] = float(scipy.optimize.fmin(g, x0=0.5, disp=False)[0])

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
myfigsize = (30,30)
#%matplotlib qt

KK = numpy.matlib.repmat(K,len(T)-1,1).reshape(161,7)
TT = numpy.matlib.repmat(T[1:],len(K),1)


fig = plt.figure(figsize=myfigsize)

ax = fig.add_subplot(111, projection='3d')

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

plt.figure(figsize = myfigsize)
#surf = ax.plot_surface(KK, TT, iv)
my_col = cm.jet(iv/np.amax(iv))
surf = ax.plot_surface(KK, TT, iv, rstride=1, cstride=1, facecolors = my_col,
        linewidth=0, antialiased=False)

ax.set_xlabel("Strikes")
ax.set_ylabel("Maturities")
ax.set_zlabel("Implied Volatility")



ax.view_init(0, -100)
plt.show()




#==============================================================================================================

import os
import json
import numpy as np
import pandas as pd
import string
import re


def CreateMatrix(folder_list, dir_path = os.path.join('P:\\JSON')):
    data_mat = []
    for folder in folder_list:
        file_list = os.listdir(os.path.join(dir_path,folder))
        file_list = [file for file in file_list if file[-4:] == ".txt"]
        if not not file_list:
            for file in file_list:
                if file[-4:] in (".txt"):
                    with open(os.path.join(dir_path,folder,file), encoding="utf8") as json_file:
                        json_Data = json.load(json_file)
                        for i in range(0,len(json_Data['messages'])):
                            data_mat.append([json_Data['messages'][i]['id'],
                                             json_Data['messages'][i]['created_at']])
    return data_mat


#============================== create matrix of all tweets ============================================================

if __name__ == '__main__':
    dir_path = os.path.join('P:\\JSON')
    folder_list = os.listdir(dir_path)                       # all folders present in directory
    data_mat = []
    data_mat = CreateMatrix(folder_list)

#with open('data_mat.pkl', 'wb' ) as f:
#    pickle.dump(data_mat, f)
data_mat = pd.read_pickle('data_mat.pkl')

#============================== convert into dataframe and save  =======================================================

#note to myself : i tried to save a pickle file for every ticker but it is was too heavy (100 mega per file..)
#so no other choice than create one huge pickle... hope RAM is enough to support


    df = pd.DataFrame(np.array(data_mat),columns=None,)
    df.columns = ['id_msg','date']
    #TODO CHECK THAT ALL DATES ARE GOOD
    #df['date'] = df['date'].astype(str).str[:10]
    df['date'] = pd.to_datetime(df['date'],format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by='date')
    df['msg'] = df['msg'].str.lower()
    df = df.reset_index(drop=True)
    df.to_pickle('P:\\df_withcorona.pkl')

    df.id_msg = df.id_msg.astype(np.int64)
    df2.id_msg = df2.id_msg.astype(np.int64)

    df = df.sort_values(by='id_msg')
    df2 = df2.sort_values(by='id_msg')
    df = df.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    test = df.id_msg - df2.id_msg
    df['date'] = df2['date']

with open('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl', 'wb' ) as f:
    pickle.dump(df.loc[45450317:,:], f)



shrcd = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Downloads', '8ksilvln99nsenn7.csv'),sep=',')
shrcd = shrcd.drop('date', axis=1)
shrcd = shrcd.drop('PERMNO', axis=1)
shrcd = shrcd.drop_duplicates()

ntwitperfirm = df[['ticker','date']].groupby('ticker').count()
ntwitperfirm = pd.merge(ntwitperfirm,shrcd, how='left', left_index=True, right_on = 'TICKER')
ntwitperfirm = ntwitperfirm.drop_duplicates(subset='ticker')
ntwitperfirm['SHRCD2'] = ntwitperfirm['SHRCD'].astype(str).str[0]

ntic_per_shrcd = ntwitperfirm[['ticker','SHRCD']].groupby('SHRCD').count()
ntic_per_shrcd2 = ntwitperfirm[['ticker','SHRCD2']].groupby('SHRCD2').count()

# Long top decile
PfWeights = RET.copy()
PfWeights.loc[:, :] = 0

for _, row in CAP.iterrows():

    col_to_keep = RET.loc[_, RET.loc[_, :] != 0].index.to_list()
    col_to_remove = RET.loc[_, RET.loc[_, :] == 0].index.to_list()
    if not row[col_to_keep].empty:
        if row.name.weekday() == 4:  # construct long short portfolio on friday 16pm
            PfWeights.loc[_, row <= np.nanpercentile(row[col_to_keep], np.arange(0, 100, 10))[0]] = -1
            PfWeights.loc[_, row >= np.nanpercentile(row[col_to_keep], np.arange(0, 100, 10))[-1]] = 1
            PfWeights.loc[_, col_to_remove] = 0
        else:
            PfWeights.loc[_, :] = np.nan

PfWeights = PfWeights.ffill()  # hold portfolio

PfWeights_unstack = PfWeights.unstack().reset_index(drop=False)
PfWeights_unstack.rename(columns={0: 'orderdirection'}, inplace=True)
sentratio_and_price = pd.merge(sentratio_and_price, PfWeights_unstack, how='left', left_on=['date', 'ticker'],
                               right_on=['date', 'ticker'])

print('Bullish events, long positioned : ',
      np.sum((sentratio_and_price['Type'] == 'Good') & (sentratio_and_price['orderdirection'] == 1)))
print('Bullish events, short positioned : ',
      np.sum((sentratio_and_price['Type'] == 'Good') & (sentratio_and_price['orderdirection'] == -1)))
print('Bullish events, no position : ',
      np.sum((sentratio_and_price['Type'] == 'Good') & (sentratio_and_price['orderdirection'] == 0)))
print('Bearish events, long positioned : ',
      np.sum((sentratio_and_price['Type'] == 'Bad') & (sentratio_and_price['orderdirection'] == 1)))
print('Bearish events, short position : ',
      np.sum((sentratio_and_price['Type'] == 'Bad') & (sentratio_and_price['orderdirection'] == -1)))
print('Bearish events, no position : ',
      np.sum((sentratio_and_price['Type'] == 'Bad') & (sentratio_and_price['orderdirection'] == 0)))
print('Neutral events, long positioned : ',
      np.sum((sentratio_and_price['Type'] == 'Neutral') & (sentratio_and_price['orderdirection'] == 1)))
print('Neutral events, short position : ',
      np.sum((sentratio_and_price['Type'] == 'Neutral') & (sentratio_and_price['orderdirection'] == -1)))
print('Neutral events, no position : ',
      np.sum((sentratio_and_price['Type'] == 'Neutral') & (sentratio_and_price['orderdirection'] == 0)))


leverage = 0
for _, row in PfWeights.iterrows():  # compute weights
    PfWeights.loc[_, row == 1] = (1 + leverage) / np.abs(np.sum(row == 1))
    PfWeights.loc[_, row == -1] = 0

PfRet_top = np.sum(RET*PfWeights.shift(1),axis=1)+1
PfRet_top = PfRet_top[PfRet_top.index<pd.to_datetime(pd.datetime(2020,1,1), utc = True)]

#long bottom decile
PfWeights = RET.copy()
PfWeights.loc[:, :] = 0

for _, row in CAP.iterrows():
    col_to_keep = RET.loc[_, RET.loc[_, :] != 0].index.to_list()
    col_to_remove = RET.loc[_, RET.loc[_, :] == 0].index.to_list()
    if not row[col_to_keep].empty:
        if row.name.weekday() == 4:  # construct long short portfolio on friday 16pm
            PfWeights.loc[_, row <= np.nanpercentile(row[col_to_keep], np.arange(0, 100, 10))[0]] = -1
            PfWeights.loc[_, row >= np.nanpercentile(row[col_to_keep], np.arange(0, 100, 10))[-1]] = 1
            PfWeights.loc[_, col_to_remove] = 0
        else:
            PfWeights.loc[_, :] = np.nan

PfWeights = PfWeights.ffill()  # hold portfolio
leverage = 1
for _, row in PfWeights.iterrows():  # compute weights
    PfWeights.loc[_, row == 1] = 0
    PfWeights.loc[_, row == -1] = leverage / np.abs(np.sum(row == -1))

PfRet_bot = np.sum(RET*PfWeights.shift(1),axis=1)+1
PfRet_bot = PfRet_bot[PfRet_bot.index<pd.to_datetime(pd.datetime(2020,1,1), utc = True)]

EquallyWeightedRet = (np.sum(RET, axis = 1)/np.sum(RET!=0, axis = 1))+1

plt.plot(PfRet_bot.index, PfRet_bot.cumprod(), 'r')
plt.plot(PfRet_top.index, PfRet_top.cumprod(), 'g')
plt.plot(EquallyWeightedRet.index, EquallyWeightedRet.cumprod(), 'b')
plt.legend(('Long Bottom Decile', 'Long Top Decile', 'Equally Weighted 19 tickers'))
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()







plt.plot(CAP.date,CAP.AAPL, 'black')
plt.scatter(sentratio_and_price.loc[(sentratio_and_price['Type'] == 'Good') & (sentratio_and_price['ticker']=='AAPL'), 'date'],
            sentratio_and_price.loc[(sentratio_and_price['Type'] == 'Good') & (sentratio_and_price['ticker']=='AAPL'), 'CAP'],
            marker='^', s=80, facecolors='none', edgecolors='g')
plt.scatter(sentratio_and_price.loc[(sentratio_and_price['Type'] == 'Neutral') & (sentratio_and_price['ticker']=='AAPL'), 'date'],
            sentratio_and_price.loc[(sentratio_and_price['Type'] == 'Neutral') & (sentratio_and_price['ticker']=='AAPL'), 'CAP'],
             s=80, facecolors='none', edgecolors='gray')
plt.scatter(sentratio_and_price.loc[(sentratio_and_price['Type'] == 'Bad') & (sentratio_and_price['ticker']=='AAPL'), 'date'],
            sentratio_and_price.loc[(sentratio_and_price['Type'] == 'Bad') & (sentratio_and_price['ticker']=='AAPL'), 'CAP'],
            marker='v', s=80, facecolors='none', edgecolors='red')
plt.show()




def portfolios(strategy, rebalancing_day,lev):
    PfWeights = RET.copy()
    PfWeights.loc[:,:] = 0

    for _, row in CAP.iterrows():

        col_to_keep = RET.loc[_,RET.loc[_,:]!= 0].index.to_list()
        col_to_remove = RET.loc[_, RET.loc[_, :] == 0].index.to_list()
        if not row[col_to_keep].empty:
            if row.name.weekday() == rebalancing_day: #construct long short portfolio on friday 16pm
                #PfWeights.loc[_, row <=  np.nanpercentile(row[col_to_keep], np.arange(0, 100, 10))[0]] = -1
                #PfWeights.loc[_, row >= np.nanpercentile(row[col_to_keep], np.arange(0, 100, 10))[-1]] = 1
                PfWeights.loc[_, row <=  -1.04] = -1
                PfWeights.loc[_, row >= 1.163] = 1
                PfWeights.loc[_, col_to_remove] = 0
            else:
                PfWeights.loc[_, :] = np.nan

    PfWeights = PfWeights.ffill() #hold portfolio

    if strategy=='Long only':
        leverage = 0
        for _, row in PfWeights.iterrows():  # compute weights
            PfWeights.loc[_, row == 1] = (1 + leverage) / np.abs(np.sum(row == 1))
            PfWeights.loc[_, row == -1] = 0
    if strategy=='Short only':
        leverage = -1
        for _, row in PfWeights.iterrows():  # compute weights
            PfWeights.loc[_, row == 1] = 0
            PfWeights.loc[_, row == -1] = leverage / np.abs(np.sum(row == -1))
    if strategy == 'Long minus short':
        leverage = lev
        for _, row in PfWeights.iterrows():  # compute weights
            PfWeights.loc[_, row == 1] = (1 + leverage) / np.sum(row == 1)
            PfWeights.loc[_, row == -1] = -leverage / np.abs(np.sum(row == -1))

    EquallyWeightedRet = np.log((np.sum(RET, axis = 1)/np.sum(RET!=0, axis = 1))+1)

    contrib = RET*PfWeights.shift(1)
    PfRet = np.log(np.sum(RET*PfWeights.shift(1),axis=1)+1)
    PfRet = PfRet[PfRet.index<pd.to_datetime(pd.datetime(2020,1,1), utc = True)]
    SPY = np.log(1+ RET['SPY'])


    plt.plot(PfRet.index,PfRet.cumsum(),'r')
    plt.plot(EquallyWeightedRet.index,EquallyWeightedRet.cumsum(),'b')
    plt.plot(RET.index,SPY.cumsum(),'g')
    plt.legend(('Polarity Portfolio','Equally-weighted Portfolio','SPY'))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.title(strategy)
    plt.show()

portfolios('Long only',0, 'noneed')
portfolios('Long only',1, 'noneed')
portfolios('Long only',2, 'noneed')
portfolios('Long only',3, 'noneed')
portfolios('Long only',4, 'noneed')
portfolios('Short only',0, 'noneed')
portfolios('Short only',1, 'noneed')
portfolios('Short only',2, 'noneed')
portfolios('Short only',3, 'noneed')
portfolios('Short only',4, 'noneed')
portfolios('Long minus short',0, 0.3)
portfolios('Long minus short',1, 0.3)
portfolios('Long minus short',2, 0.3)
portfolios('Long minus short',3, 0.3)
portfolios('Long minus short',4, 0.01)
portfolios('Long minus short',4, 0.3)
portfolios('Long minus short',4, 1)


def confmat(data,strat, bot_thresh, top_thresh, rebalanceday,pr='no'):
    #Long top decile
    PfWeights = RET.copy()
    PfWeights.loc[:, :] = 0

    for _, row in CAP.iterrows():

        col_to_keep = RET.loc[_, RET.loc[_, :] != 0].index.to_list()
        col_to_remove = RET.loc[_, RET.loc[_, :] == 0].index.to_list()
        if not row[col_to_keep].empty:
            if row.name.weekday() == rebalanceday:  # construct long short portfolio on friday 16pm
                PfWeights.loc[_, row >= top_thresh] = 1
                PfWeights.loc[_, row < bot_thresh] = -1
                PfWeights.loc[_, col_to_remove] = 0
            else:
                PfWeights.loc[_, :] = np.nan

    PfWeights = PfWeights.ffill()  # hold portfolio

    PfWeights_unstack = PfWeights.unstack().reset_index(drop=False)
    PfWeights_unstack.rename(columns={0:'orderdirection'},inplace=True)
    data = pd.merge(data, PfWeights_unstack,  how='left', left_on=['date','ticker'], right_on = ['date','ticker'])

    measures = dict()

    Bull_Long = np.sum((data['Type'] == 'Good') & (data['orderdirection'] == 1))
    Bull_Short = np.sum((data['Type'] == 'Good') & (data['orderdirection'] == -1))
    Bull_None = np.sum((data['Type'] == 'Good') & (data['orderdirection'] == 0))
    Bear_Long = np.sum((data['Type'] == 'Bad') & (data['orderdirection'] == 1))
    Bear_Short = np.sum((data['Type'] == 'Bad') & (data['orderdirection'] == -1))
    Bear_None = np.sum((data['Type'] == 'Bad') & (data['orderdirection'] == 0))
    Neut_Long = np.sum((data['Type'] == 'Neutral') & (data['orderdirection'] == 1))
    Neut_Short = np.sum((data['Type'] == 'Neutral') & (data['orderdirection'] == -1))
    Neut_None = np.sum((data['Type'] == 'Neutral') & (data['orderdirection'] == 0))
    NoEvent_Long = np.sum((data['Event3'] == False) & (data['orderdirection'] == 1))
    NoEvent_Short = np.sum((data['Event3'] == False) & (data['orderdirection'] == -1))
    NoEvent_None = np.sum((data['Event3'] == False) & (data['orderdirection'] == 0))

    if strat == "long":
        measures['TP'] = Bull_Long
        measures['FN'] = Bull_Short + Neut_Short + NoEvent_Short
        measures['FP'] = Bear_Long + Neut_Long + NoEvent_Long
        measures['TN'] = Bear_Short

    if strat == "short":
        measures['TP'] = Bear_Short
        measures['FN'] = Bear_Long + Neut_Long + NoEvent_Long
        measures['FP'] = Bull_Short + Neut_Short + NoEvent_Short
        measures['TN'] = Bull_Long

    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    measures['f1score'] = 2 * measures['precision'] * measures['recall'] / (measures['precision'] + measures['recall'])

    if pr=='yes':
        print('-'*15, 'rebalance day : ', rebalanceday, '-'*15)
        print('Bullish events, long positioned : ', Bull_Long)
        print('Bullish events, short positioned : ', Bull_Short)
        print('Bullish events, no position : ', Bull_None)
        print('Bearish events, long positioned : ',Bear_Long)
        print('Bearish events, short position : ', Bear_Short)
        print('Bearish events, no position : ', Bear_None)
        print('Neutral events, long positioned : ', Neut_Long)
        print('Neutral events, short position : ', Neut_Short)
        print('Neutral events, no position : ', Neut_None)
        print('No events, no position : ', NoEvent_None)
        print('No events, long position : ', NoEvent_Long)
        print('No events, short position : ', NoEvent_Short)
    return measures

maxs = []
for reb in [0,1,2,3,4]:
    obs1  = []
    for thresh in  np.linspace(-3,3,50):
        measures = confmat(sentratio_and_price, 'long', thresh,thresh, reb,pr='no')
        obs1.append([thresh, measures['f1score']])

    obs2  = []
    for thresh in  np.linspace(-3,3,50):
        measures = confmat(sentratio_and_price, 'short', thresh,thresh, reb,pr='no')
        obs2.append([thresh, measures['f1score']])

    max1 = max(obs1,key=lambda x:x[1])
    max2 = max(obs2,key=lambda x:x[1])

    if reb==0:
        day='Monday'
        freqpar = 'W-MON'
    if reb==1:
        day='Tuesday'
        freqpar = 'W-TUE'
    if reb==2:
        day='Wednesday'
        freqpar = 'W-WED'
    if reb==3:
        day ='Thursday'
        freqpar = 'W-THU'
    if reb==4:
        day ='Friday'
        freqpar = 'W-FRI'

    plt.style.use('seaborn-whitegrid')
    plt.plot([item[0] for item in obs1], [item[1] for item in obs1], '-', color='green',label ='Long')
    plt.plot([item[0] for item in obs2], [item[1] for item in obs2], '-', color='red', label='Short')
    plt.scatter(max2[0], max2[1], s=80, facecolors='none', edgecolors='blue')
    plt.scatter(max1[0], max1[1], s=80, facecolors='none', edgecolors='blue')
    #plt.axvline(x=max2[0],  ymax=0.931, color = 'blue',  linestyle='--')
    #plt.axvline(x=max1[0], ymax= 0.921, color = 'blue',  linestyle='--')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title('Optimal CAP thresholds, rebalancing day : ' + day)
    #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
    plt.show()

    confmat(sentratio_and_price, 'long', max1[0],max2[0], reb,pr='yes')
    print(day,' : ', max1[0],' ',max2[0])
    maxs.append([reb,max1[0],max2[0]])




def confmat2(data,strat, bot_thresh, top_thresh, rebalanceday,pr='no'):
    PfWeights = RET.copy()
    PfWeights.loc[:, :] = 0
    if reb == 0:
        day = 'Monday'
        freqpar = 'W-MON'
    if reb == 1:
        day = 'Tuesday'
        freqpar = 'W-TUE'
    if reb == 2:
        day = 'Wednesday'
        freqpar = 'W-WED'
    if reb == 3:
        day = 'Thursday'
        freqpar = 'W-THU'
    if reb == 4:
        day = 'Friday'
        freqpar = 'W-FRI'

    for _, row in CAP.iterrows():
        col_to_keep = RET.loc[_, RET.loc[_, :] != 0].index.to_list()
        col_to_remove = RET.loc[_, RET.loc[_, :] == 0].index.to_list()
        if not row[col_to_keep].empty:
            if row.name.weekday() == rebalanceday:  # construct long short portfolio on friday 16pm
                PfWeights.loc[_, row >= top_thresh] = 1
                PfWeights.loc[_, row < bot_thresh] = -1
                PfWeights.loc[_, col_to_remove] = 0
            else:
                PfWeights.loc[_, :] = np.nan

    data = data[data['date'] > pd.to_datetime(pd.datetime(2011,1,21), utc = True)]
    tickers = sentratio_and_price['ticker'].unique().tolist()
    uniques = [pd.date_range(start=sentratio_and_price['date'].min(), end=sentratio_and_price['date'].max(), freq=freqpar).tolist(), tickers, ['Good','Neutral','Bad']]   #todo start='2012-01-01', end='2020-03-23'
    cadre = pd.DataFrame(product(*uniques), columns = ['date','ticker','Type'])
    cadre = cadre.sort_values(['ticker', 'date'], ascending=[1, 1])
    see = sentratio_and_price.set_index('date').groupby([pd.Grouper(freq=freqpar),'ticker','Type'])['Event3'].count().reset_index().sort_values(['date', 'ticker'], ascending=[1, 1])
    see['date'] = see['date'] + timedelta(days=-7)
    see = pd.merge(cadre, see,  how='left', left_on=['date','ticker','Type'], right_on = ['date','ticker','Type'])

    perweek = pd.pivot_table(see, values = 'Event3', index = ['date','ticker'], columns='Type',dropna = False).reset_index()
    PfWeights_unstack = PfWeights.unstack().reset_index(drop=False)
    PfWeights_unstack.rename(columns={0: 'orderdirection'}, inplace=True)
    perweek = pd.merge(perweek, PfWeights_unstack, how='left', left_on=['date', 'ticker'], right_on=['date', 'ticker'])

    #does not sum to number of weeks because multiple events can happen on the same week
    # not better to 3 class pb?
    # neutral & no direction is correct
    Bull_Long = np.sum((perweek['Good'] > 0) & (perweek['orderdirection'] == 1))
    Bull_Short = np.sum((perweek['Good'] > 0) & (perweek['orderdirection'] == -1))
    Bull_None = np.sum((perweek['Good'] > 0) & (perweek['orderdirection'] == 0))
    Bear_Long = np.sum((perweek['Bad'] > 0) & (perweek['orderdirection'] == 1))
    Bear_Short = np.sum((perweek['Bad'] > 0) & (perweek['orderdirection'] == -1))
    Bear_None = np.sum((perweek['Bad'] > 0) & (perweek['orderdirection'] == 0))
    Neut_Long = np.sum((perweek['Neutral'] > 0) & (perweek['orderdirection'] == 1))
    Neut_Short = np.sum((perweek['Neutral'] > 0) & (perweek['orderdirection'] == -1))
    Neut_None = np.sum((perweek['Neutral'] > 0) & (perweek['orderdirection'] == 0))
    NoEvent_Long = np.sum((perweek['Good'].isna() & perweek['Neutral'].isna() & perweek['Bad'].isna()) & (perweek['orderdirection'] == 1))
    NoEvent_Short = np.sum((perweek['Good'].isna() & perweek['Neutral'].isna() & perweek['Bad'].isna()) & (perweek['orderdirection'] == -1))
    NoEvent_None = np.sum((perweek['Good'].isna() & perweek['Neutral'].isna() & perweek['Bad'].isna()) & (perweek['orderdirection'] == 0))

    measures = dict()
    if strat == "long":
        measures['TP'] = Bull_Long
        measures['FN'] = Bull_Short + Neut_Short + NoEvent_Short
        measures['FP'] = Bear_Long + Neut_Long + NoEvent_Long
        measures['TN'] = Bear_Short

    if strat == "short":
        measures['TP'] = Bear_Short
        measures['FN'] = Bear_Long + Neut_Long + NoEvent_Long
        measures['FP'] = Bull_Short + Neut_Short + NoEvent_Short
        measures['TN'] = Bull_Long

    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    measures['f1score'] = 2 * measures['precision'] * measures['recall'] / (measures['precision'] + measures['recall'])

    if pr == 'yes':
        print('-' * 15, 'rebalance day : ', rebalanceday, '-' * 15)
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
    return measures


maxs = []
for reb in [0,1,2,3,4]:
    obs1  = []
    for thresh in  np.linspace(-3,3,30):
        measures = confmat2(sentratio_and_price, 'long', thresh,thresh, reb,pr='no')
        obs1.append([thresh, measures['f1score']])

    obs2  = []
    for thresh in  np.linspace(-3,3,30):
        measures = confmat2(sentratio_and_price, 'short', thresh,thresh, reb,pr='no')
        obs2.append([thresh, measures['f1score']])

    max1 = max(obs1,key=lambda x:x[1])
    max2 = max(obs2,key=lambda x:x[1])

    if reb==0:
        day='Monday'
        freqpar = 'W-MON'
    if reb==1:
        day='Tuesday'
        freqpar = 'W-TUE'
    if reb==2:
        day='Wednesday'
        freqpar = 'W-WED'
    if reb==3:
        day ='Thursday'
        freqpar = 'W-THU'
    if reb==4:
        day ='Friday'
        freqpar = 'W-FRI'

    plt.style.use('seaborn-whitegrid')
    plt.plot([item[0] for item in obs1], [item[1] for item in obs1], '-', color='green',label ='Long')
    plt.plot([item[0] for item in obs2], [item[1] for item in obs2], '-', color='red', label='Short')
    plt.scatter(max2[0], max2[1], s=80, facecolors='none', edgecolors='blue')
    plt.scatter(max1[0], max1[1], s=80, facecolors='none', edgecolors='blue')
    #plt.axvline(x=max2[0],  ymax=0.931, color = 'blue',  linestyle='--')
    #plt.axvline(x=max1[0], ymax= 0.921, color = 'blue',  linestyle='--')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title('Optimal CAP thresholds, rebalancing day : ' + day)
    #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
    plt.show()

    confmat2(sentratio_and_price, 'long', max1[0],max2[0], reb,pr='yes')
    print(day,' : ', max1[0],' ',max2[0])
    maxs.append([reb,max1[0],max2[0]])



def power_utility(w, gamma):
    #gamma>0  : risk averse
    #gamma<0 : risk seeker
    #gamma = 1 : log utility
    return ( w**(1-gamma) -1 ) / (1-gamma)

initial_W = 100

PFS_W = intial_W*(1+PFS)
PFS_Utility.apply(power_utility(PFS_W, gamma = 0.5)) #risk averse


#sns.displot(PFS.reset_index().melt(id_vars='date', var_name='type', value_name='ret'), x = 'ret' , hue = 'type' , kind = 'kde')
#plt.show()


#sns.displot(PFS.reset_index().melt(id_vars='date', var_name='type', value_name='ret'), x = 'ret' , hue = 'type' , element = 'step')
#plt.show()

#count, bins_count_short = np.histogram(PFS['short'].dropna(),100)
#pdf_short = count / sum(count)
#cdf_short = np.cumsum(pdf_short)
#count, bins_count_long = np.histogram(PFS['long'].dropna(),100)
#pdf_long = count / sum(count)
#cdf_long = np.cumsum(pdf_long)

## plotting PDF and CDF
#plt.plot(bins_count_short[1:], cdf_short, label="short")
#plt.plot(bins_count_long[1:], cdf_long, label="long")
#plt.legend()
#plt.show()


#daily portfolio
plt.plot(PFS.index, PFS.loc[:, 'long'].apply(perf).fillna(0).cumsum(), 'r')
plt.plot(PFS.index, PFS.loc[:, 'short'].apply(perf).fillna(0).cumsum(), 'b')
plt.legend(('Long Polarity Portfolio', 'Short Polarity Portfolio'))
plt.xlabel('Date')
plt.ylabel('Cumulative Log Return')
plt.title('long = ' + str(long) + ', short = ' + str(short))
plt.show()

#weekly portfolio
weekday = ['Monday', 'Tuesday','Wednesday','Thursday','Friday']
for i in range(5):
    print(i)
    plt.plot(PFS.loc[PFS.index.weekday == i].index, PFS.loc[PFS.index.weekday == i, 'long'].apply(perf).fillna(0).cumsum(), 'r')
    plt.plot(PFS.loc[PFS.index.weekday == i].index, PFS.loc[PFS.index.weekday == i, 'short'].apply(perf).fillna(0).cumsum(), 'b')
    plt.legend(('Long Polarity Portfolio', 'Short Polarity Portfolio'))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Portfolio')
    plt.title(weekday[i])
    plt.show()



shortspan = [ 0.        , -0.76389114, -1.52778229, -2.29167343, -3,-3.5, -4,
       -3.81945572, -4.58334687]
longspan = [ 0.        ,  1.19876105,  2.39752211, 3, 3.5, 4,  4.79504421,
        5.99380526,  7.19256632,  8]


Ps = []
for short in np.linspace(0, sentratio_and_price['CAP'].min() + 0.5, 10):    #np.linspace(0, sentratio_and_price['CAP'].min() + 0.5, 10)
    for long in np.linspace(0,sentratio_and_price['CAP'].max()-0.5,10):  #np.linspace(0,sentratio_and_price['CAP'].max()-0.5,10)
        if long > short:
            #max_short[0]: maximizes f1 short   #max_long[0] : max f1 long, max_prec_long[0] : max precision long, max_prec_short[0] : max precision short
            sentratio_and_price['opti_position'] = sentratio_and_price['CAP'].apply(pos,short_thresh = short,long_thresh = long)
            sentratio_and_price['long_position'] = sentratio_and_price['CAP'].apply(longpos,long_thresh = long)
            sentratio_and_price['short_position'] = sentratio_and_price['CAP'].apply(shortpos,short_thresh = short)

            sentratio_and_price.loc[sentratio_and_price['excess_return'].isna(), 'opti_position'] = 0
            sentratio_and_price.loc[sentratio_and_price['excess_return'].isna(), 'long_position'] = 0
            sentratio_and_price.loc[sentratio_and_price['excess_return'].isna(), 'short_position'] = 0

            OPTIPOS = pd.pivot_table(sentratio_and_price, values = 'opti_position', index = 'date', columns='ticker',dropna = False)
            LONGPOS = pd.pivot_table(sentratio_and_price, values = 'long_position', index = 'date', columns='ticker',dropna = False)
            SHORTPOS = pd.pivot_table(sentratio_and_price, values = 'short_position', index = 'date', columns='ticker',dropna = False)

            EXCESSRET = pd.pivot_table(sentratio_and_price, values = 'excess_return', index = 'date', columns='ticker',dropna = False)
            EXCESSRET = EXCESSRET.replace(np.nan, 0)

            holding_period = 1
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=holding_period)
            EXCESSRET_next5 =(1.+ EXCESSRET).shift(-1).rolling(window=indexer).agg(lambda x : x.prod()) -1

            PF_LONG = np.sum(LONGPOS * EXCESSRET_next5 , axis = 1) / np.sum(LONGPOS!=0, axis = 1)
            PF_SHORT = np.sum(SHORTPOS * EXCESSRET_next5 , axis = 1) / np.sum(SHORTPOS!=0, axis = 1)

            PFS = pd.merge(pd.DataFrame({'long':PF_LONG}), pd.DataFrame({'short':PF_SHORT}), how='left',right_index=True,left_index=True)
            U, p_brunnermunzel = scipy.stats.brunnermunzel(PFS['long'].dropna(), PFS['short'].dropna())
            n1 = PFS['long'].count()
            n2 = PFS['short'].count()
            #Z = (int(U) - int(n1) * int(n2) / 2) / np.sqrt(int(n1) * int(n2) * (int(n1) + int(n2) + 1) / 12)

            x1bar, x2bar, std1, std2 = np.nanmean(PFS['long']) , np.nanmean(PFS['short']) ,  np.nanstd(PFS['long']) ,  np.nanstd(PFS['short'])
            tstat = (x1bar - x2bar) / (np.sqrt(((n1 - 1)*std1**2 + (n2-1)*std2**2) /(n1+n2-2)*(1/n1 + 1/n2)))
            _ , p_mean2 = scipy.stats.ttest_ind(PFS['long'], PFS['short'], axis=None, equal_var=False, nan_policy='omit')
            p_mean = 1 - t.cdf(tstat, n1+n2-2)
            Ps.append([short,long,p_brunnermunzel, p_mean2])

            folder = "long " + str("{0:.2f}".format(long)) + ', short ' + str("{0:.2f}".format(short))
            path = os.path.join(os.getcwd(), 'results', 'Portfolios','Static Thresholds' ,'CAP9','Reset', folder)
            if not os.path.exists(path):
                os.makedirs(path)

            plt.plot(PF_LONG.index, PF_LONG, color='green', label='Long')
            plt.plot(PF_SHORT.index, PF_SHORT, color='red', label='Short')
            plt.legend()
            plt.title(str(holding_period) + '-days portfolio returns, long = ' + str(long) + ', short = ' + str(short))
            plt.savefig(os.path.join(path, 'returns.jpg'))
            #plt.show()
            plt.close()

            ax = sns.boxplot(data=PFS, whis=1.5, palette={"long": "green", "short": "red"})
            plt.title(str(holding_period) + '-days portfolio returns, long = ' + str(long) + ', short = ' + str(short))
            # plt.ylim([-0.1,0.1])
            plt.savefig(os.path.join(path, 'boxplot.jpg'))
            #plt.show()
            plt.close()

            ax = sns.boxplot(data=PFS, whis=1.5, palette={"long": "green", "short": "red"})
            plt.title(str(holding_period) + '-days portfolio returns, long = ' + str(long) + ', short = ' + str(short))
            plt.ylim([-0.1, 0.1])
            plt.savefig(os.path.join(path, 'boxplot_zoomed.jpg'))
            #plt.show()
            plt.close()

            keep = np.sum(RET, axis=1) != 0
            plt.plot(LONGPOS.loc[keep, :].index, np.sum(LONGPOS.loc[keep, :] != 0, axis=1), color='green', label='Long')
            plt.plot(SHORTPOS.loc[keep, :].index, np.sum(SHORTPOS.loc[keep, :] != 0, axis=1), color='red',
                     label='Short')
            plt.legend()
            plt.title('Number of positions, long = ' + str(long) + ', short = ' + str(short))
            plt.savefig(os.path.join(path, 'positions.jpg'))
            #plt.show()
            plt.close()

            plt.hist(np.sum(LONGPOS != 0, axis=1), bins=np.max(np.sum(LONGPOS != 0, axis=1)))
            plt.title('Number of positions - Long portfolio, long = ' + str(long))
            plt.savefig(os.path.join(path, 'distribution_long_positions.jpg'))
            #plt.show()
            plt.close()

            plt.hist(np.sum(SHORTPOS != 0, axis=1), bins=np.max(np.sum(SHORTPOS != 0, axis=1)))
            plt.title('Histogram - Number of positions - Short portfolio, short = ' + str(short))
            plt.savefig(os.path.join(path, 'distribution_short_positions.jpg'))
            #plt.show()
            plt.close()

            # daily portfolio  #-3   5  looks ok
            plt.plot(PFS.index, PFS.loc[:, 'long'].apply(perf).fillna(0).cumsum(), 'g')
            plt.plot(PFS.index, PFS.loc[:, 'short'].apply(perf).fillna(0).cumsum(), 'r')
            plt.plot(sentratio_and_price.loc[sentratio_and_price['ticker']=='AAPL', 'date'], sentratio_and_price.loc[sentratio_and_price['ticker']=='AAPL', 'market_return'].apply(perf).fillna(0).cumsum(), 'b')
            plt.legend(('Long Polarity Portfolio', 'Short Polarity Portfolio', 'SPY'))
            plt.xlabel('Date')
            plt.ylabel('Cumulative Log Return')
            plt.title('long = ' + str(long) + ', short = ' + str(short))
            plt.savefig(os.path.join(path , 'cumulative_log_return.jpg'))
            #plt.show()
            plt.close()


scipy.stats.ttest_ind(PFS['long'], PFS['short'], axis=None, equal_var=False, nan_policy='omit')



Ps = pd.DataFrame(Ps, columns = ['short threshold', 'long threshold', 'p_brunnermunzel', 'p_ttest'])
Ps = Ps.round(3)

sns.heatmap(Ps.pivot(index='short threshold', columns='long threshold', values='p_brunnermunzel'))
plt.title('Brunnemunzel p-value heatmap')
plt.savefig(os.path.join(os.getcwd(), 'results', 'Portfolios','CAP9' ,'Reset', 'Brunnemunzel_heatmap.jpg'))
plt.show()
plt.close()

sns.heatmap(Ps.pivot(index='short threshold', columns='long threshold', values='p_ttest'))
plt.title('Mean test p-value heatmap')
plt.savefig(os.path.join(os.getcwd(), 'results', 'Portfolios','CAP9' , 'Reset','mean_test_heatmap.jpg'))
plt.show()
plt.close()

# benchmarking portfolios

long_reg = pd.merge(pd.DataFrame(PF_LONG),sentratio_and_price.loc[sentratio_and_price['ticker']=='AAPL',['date','market_return','t90ret']],left_index=True,right_on='date')
long_reg.rename(columns={0:'long_pf_ret'}, inplace=True)
long_reg = long_reg.fillna(0)
long_reg.loc[:,'long_pf_ret'] = np.log(1+long_reg.loc[:,'long_pf_ret'])
long_reg.loc[:,'market_return'] = np.log(1+long_reg.loc[:,'market_return']-long_reg.loc[:,'t90ret'])

reg = sm.OLS(long_reg['long_pf_ret'], sm.add_constant(long_reg['market_return']), missing='drop').fit()
par = reg.params
sum = reg.summary2()
print(sum)

short_reg = pd.merge(pd.DataFrame(PF_SHORT),sentratio_and_price.loc[sentratio_and_price['ticker']=='AAPL',['date','market_return','t90ret']],left_index=True,right_on='date')
short_reg.rename(columns={0:'short_pf_ret'}, inplace=True)
short_reg = short_reg.fillna(0)
short_reg.loc[:,'short_pf_ret'] = np.log(1+short_reg.loc[:,'short_pf_ret'])
short_reg.loc[:,'market_return'] = np.log(1+short_reg.loc[:,'market_return']-short_reg.loc[:,'t90ret'])

reg = sm.OLS(short_reg['short_pf_ret'], sm.add_constant(short_reg['market_return']), missing='drop').fit()
par = reg.params
sum = reg.summary2()
print(sum)

# fama french factors

fff = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\tweet_sklearn\\F-F_Research_Data_Factors_daily.csv",sep=',')
fff.rename(columns={'Unnamed: 0':'date'},inplace=True)
fff['date'] = pd.to_datetime(fff['date'],format="%Y%m%d")
fff['date'] = pd.to_datetime(fff['date'], utc = True)

fff[['Mkt-RF','SMB','HML']] = fff[['Mkt-RF','SMB','HML']]/100

long_reg = pd.merge(pd.DataFrame(PF_LONG),fff[['date','Mkt-RF','SMB','HML']],left_index=True,right_on='date')
long_reg.rename(columns={0:'long_pf_ret'}, inplace=True)
long_reg = long_reg.fillna(0)
long_reg.loc[:,'long_pf_ret'] = np.log(1+long_reg.loc[:,'long_pf_ret'])

reg = sm.OLS(long_reg['long_pf_ret'], sm.add_constant(long_reg[['Mkt-RF','SMB','HML']]), missing='drop').fit()
par = reg.params
sum = reg.summary2()
print(sum)

short_reg = pd.merge(pd.DataFrame(PF_SHORT),fff[['date','Mkt-RF','SMB','HML']],left_index=True,right_on='date')
short_reg.rename(columns={0:'short_pf_ret'}, inplace=True)
short_reg = short_reg.fillna(0)
short_reg.loc[:,'short_pf_ret'] = np.log(1+short_reg.loc[:,'short_pf_ret'])

reg = sm.OLS(short_reg['short_pf_ret'], sm.add_constant(short_reg[['Mkt-RF','SMB','HML']]), missing='drop').fit()
par = reg.params
sum = reg.summary2()
print(sum)

