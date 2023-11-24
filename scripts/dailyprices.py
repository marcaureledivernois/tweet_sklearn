import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import seaborn as sns
import statsmodels.api as sm
import os
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS
import datetime

#df = pd.read_pickle('df_clean.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_1.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_2.pkl')
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl')])  #sent with opti thresholds

#sentratio_and_price = pd.read_pickle('sentratio_and_price.pkl')

#=========================== use classified tweets =========================================

#sentiment ratio aggregated on a weekly basis (end=wednesday) VS daily return of thursday

def bullishness(x):
    bull = x['sent_merged'] == 1
    bear = x['sent_merged'] == -1
    neutral = x['sent_merged'] == 0
    return x.loc[bull, 'sent_merged'].count() / (x.loc[bull, 'sent_merged'].sum() + x.loc[neutral, 'sent_merged'].shape[0]
                                               + np.abs(x.loc[bear, 'sent_merged'].sum()))

def neutral(x):
    bull = x['sent_merged'] == 1
    bear = x['sent_merged'] == -1
    neutral = x['sent_merged'] == 0
    return x.loc[neutral, 'sent_merged'].count() / (x.loc[bull, 'sent_merged'].sum() + x.loc[neutral, 'sent_merged'].shape[0]
                                               + np.abs(x.loc[bear, 'sent_merged'].sum()))

def bearishness(x):
    bull = x['sent_merged'] == 1
    bear = x['sent_merged'] == -1
    neutral = x['sent_merged'] == 0
    return x.loc[bear, 'sent_merged'].count() / (x.loc[bull, 'sent_merged'].sum() + x.loc[neutral, 'sent_merged'].shape[0]
                                               + np.abs(x.loc[bear, 'sent_merged'].sum()))


sentratio = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W-WED'),'ticker']).apply(bullishness))
sentratio.rename(columns={0:'Bullishness'}, inplace=True)
sentratio['Neutral'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W-WED'),'ticker']).apply(neutral))
sentratio['Bearishness'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W-WED'),'ticker']).apply(bearishness))
sentratio['N'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W-WED'),'ticker'])['sent_merged'].count())['sent_merged']
sentratio = sentratio.reset_index()



#============================== merge sentiment and market var =========================================================#

matching_table = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'table_de_corr_changetickers.csv'),sep=';')

#correction is done slightly below

# accounting : accounting data
accounting = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\extended_db.csv",sep=';')
accounting['datadate'] = pd.to_datetime(accounting['datadate'],format="%Y%m%d")
accounting['Qdate'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in accounting.datadate]

prices_db = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\tweet_sklearn\\daily_crsp.csv",sep=',')
prices_db.TICKER = prices_db.TICKER.replace(matching_table.set_index('old').new.to_dict()) #correct tickers with matching table
prices_db['date'] = pd.to_datetime(prices_db['date'],format="%Y%m%d")
prices_db['adjprice'] = prices_db['PRC']/prices_db['CFACPR']
prices_db = prices_db.sort_values(['TICKER', 'date'], ascending=[1, 1])
prices_db['daily_return'] = prices_db.groupby('TICKER').adjprice.pct_change()
prices_db['daily_return_t+1'] =  prices_db.groupby(['TICKER'])['daily_return'].shift(-1)
prices_db['VOL_t+1'] =  prices_db.groupby(['TICKER'])['VOL'].shift(-1)

def getlast(df):
    return df.iloc[-1]

ret = pd.DataFrame(prices_db.set_index('date').groupby([pd.Grouper(freq='W-WED'),'TICKER'])['adjprice'].apply(getlast))
ret = ret.reset_index()
ret = ret.sort_values(['TICKER', 'date'], ascending=[1, 1])
ret['weekly_adjprice_t+1'] =  ret.groupby(['TICKER'])['adjprice'].shift(-1)
ret['weekly_adjprice_t-1'] =  ret.groupby(['TICKER'])['adjprice'].shift(1)
ret['monthly_adjprice_t+1'] =  ret.groupby(['TICKER'])['adjprice'].shift(-4)
ret['Monthly_return_t+1'] =  ret['monthly_adjprice_t+1']/ret['adjprice']-1
ret['Weekly_return_t'] =  ret['adjprice']/ret['weekly_adjprice_t-1']-1
ret['Weekly_return_t+1'] =  ret['weekly_adjprice_t+1']/ret['adjprice']-1
ret = ret.drop(['adjprice','monthly_adjprice_t+1','weekly_adjprice_t+1','weekly_adjprice_t-1'], axis=1)

prices_db = pd.merge(prices_db, ret,  how='left', left_on=['date','TICKER'], right_on = ['date','TICKER'])


mean_weekly_volume = pd.DataFrame(prices_db.set_index('date').groupby([pd.Grouper(freq='W-WED'),'TICKER'])['VOL'].mean())
mean_weekly_volume = mean_weekly_volume.reset_index()
mean_weekly_volume.rename(columns={'VOL':'mean_weekly_vol'}, inplace=True)

sentratio_and_price = pd.merge(sentratio, prices_db,  how='left', left_on=['date','ticker'], right_on = ['date','TICKER'])

sentratio_and_price = pd.merge(sentratio_and_price, mean_weekly_volume,  how='left', left_on=['date','ticker'], right_on = ['date','TICKER'])
sentratio_and_price = sentratio_and_price.drop(['PERMNO', 'PERMCO','ISSUNO','CUSIP','RET','CFACPR','CFACSHR', 'ticker', 'TICKER_y'], axis=1)
sentratio_and_price.rename(columns={'TICKER_x':'TICKER'}, inplace=True)

#============================== LAGGED VAR =========================================================#

#TODO THIS IS PROBABLY WRONG. AFTER THE JOIN, SOME DATES DISAPPER SO I CANT USE DIFF OR SHIFT...!
sentratio_and_price['mean_weekly_vol_t+1'] =  sentratio_and_price.groupby(['TICKER'])['mean_weekly_vol'].shift(-1)

sentratio_and_price['D_Bullishness'] = np.abs(sentratio_and_price.groupby(['TICKER'])['Bullishness'].diff())
sentratio_and_price['D_VOL_t+1'] = sentratio_and_price.groupby(['TICKER'])['VOL_t+1'].diff()
sentratio_and_price['D_VOL'] = sentratio_and_price.groupby(['TICKER'])['VOL'].diff()
sentratio_and_price['D_mean_weekly_vol'] =  sentratio_and_price.groupby(['TICKER'])['mean_weekly_vol'].diff()
sentratio_and_price['D_mean_weekly_vol_t+1'] =  sentratio_and_price.groupby(['TICKER'])['D_mean_weekly_vol'].shift(-1)
sentratio_and_price['D_N'] =  sentratio_and_price.groupby(['TICKER'])['N'].diff()

#Bid-ask spreads usually widen in highly volatile environments
sentratio_and_price['spread'] = sentratio_and_price['ASK'] - sentratio_and_price['BID']
sentratio_and_price['spread_t+1'] = sentratio_and_price.groupby(['TICKER'])['spread'].shift(-1)
sentratio_and_price['D_spread'] = sentratio_and_price.groupby(['TICKER'])['spread'].diff()
#============================== boxplot drawdowns =========================================================#


def test_distrib(thresh,y_var,return_to_pred,N_mini,full_db, jmptype):
    full_db_copy = full_db.copy()
    full_db_copy = full_db_copy.drop(full_db_copy[full_db_copy.N < N_mini].index)

    if jmptype == "Drawdown":
        full_db_copy['dummy'] = (full_db_copy[return_to_pred] < thresh) * 1
    elif jmptype == "Drawup":
        full_db_copy['dummy'] = (full_db_copy[return_to_pred] > thresh) * 1
    elif jmptype == "Combined":
        full_db_copy['dummy'] = (np.abs(full_db_copy[return_to_pred]) > thresh) * 1
    else:
        print("Incorrect jmptype")

    with pd.option_context('mode.use_inf_as_na', True):
        full_db_copy = full_db_copy.dropna().reset_index(drop=True)

    ax = sns.boxplot(x="dummy", y=y_var, data=full_db_copy, whis=1.5)
    #ax = sns.stripplot(x='drawdown_dummy', y=y_var, data=full_db_copy, color="orange", jitter=0.03, size=1.5)
    plt.title('return: ' + return_to_pred + ', threshold: ' + str(thresh) + ', N_mini ' + str(N_mini))
    plt.ylabel(y_var + " Ratio")
    plt.xlabel(jmptype +" dummy")
    if return_to_pred[-3:] == "t+1":
        plt.savefig('boxplot_' + jmptype + '_' + y_var + '_' + return_to_pred[:3] + '2' + '.jpg')
    else:
        plt.savefig('boxplot_'  + jmptype + '_' + y_var + '_' + return_to_pred[:3] + '.jpg')
    plt.show()

thresh = -0.05
N_mini = 50

test_distrib(thresh,"Bearishness",'daily_return',N_mini,sentratio_and_price, "Drawdown")
test_distrib(thresh,"Bearishness",'daily_return_t+1',N_mini,sentratio_and_price, "Drawdown")
test_distrib(thresh,"Bearishness",'Weekly_return_t',N_mini,sentratio_and_price, "Drawdown")
test_distrib(thresh,"Bearishness",'Weekly_return_t+1',N_mini,sentratio_and_price, "Drawdown")
test_distrib(thresh,"Bearishness",'Monthly_return_t+1',N_mini,sentratio_and_price, "Drawdown")


#============================== reg volume vs Bullishness =========================================================#

def reg_vol(data,N_mini):
    sentratio_and_price_copy = data.copy()
    with pd.option_context('mode.use_inf_as_na', True):
        sentratio_and_price_copy = sentratio_and_price_copy.dropna().reset_index(drop=True)

    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.N < N_mini].index)
    X = sm.add_constant(sentratio_and_price_copy[['D_Bullishness']])  # --- with constant

    y = sentratio_and_price_copy['D_mean_weekly_vol']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = OLS(yTrain, xTrain)
    res = model.fit()

    xplot = sentratio_and_price_copy['D_Bullishness'].values
    yplot = sentratio_and_price_copy['D_mean_weekly_vol'].values
    plt.scatter(xplot,yplot)
    plt.xlabel('D_Bullishness')
    plt.ylabel('D_mean_weekly_vol')
    plt.show()
    print(res.summary())

reg_vol(sentratio_and_price,200)   #counter intuitive result


#============================== reg volume vs Number tweets posted =====================================================#

def reg_vol_N(data,N_mini,startyear):
    sentratio_and_price_copy = data.copy()
    with pd.option_context('mode.use_inf_as_na', True):
        sentratio_and_price_copy = sentratio_and_price_copy.dropna().reset_index(drop=True)

    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.date < pd.to_datetime(datetime(startyear, 1, 1), utc = True) ].index)
    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.N < N_mini].index)
    X = sm.add_constant(sentratio_and_price_copy[['D_N']])  # --- with constant

    y = sentratio_and_price_copy['D_mean_weekly_vol']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = OLS(yTrain, xTrain)
    res = model.fit()

    xplot = sentratio_and_price_copy['D_N'].values
    yplot = sentratio_and_price_copy['D_mean_weekly_vol'].values
    plt.scatter(xplot,yplot)
    plt.xlabel('Change in number of tweets posted per week')
    plt.ylabel('Change in average weekly volume of transactions')
    plt.title('Activity vs Volume of Transaction')
    plt.grid(b='blue')
    #plt.savefig('activity-volume.jpg')
    plt.show()
    print(res.summary())

reg_vol_N(sentratio_and_price,20,2012) #shows nicely that when ppl post more, the volume of transactions increases
#todo this motivates that ppl tweet on the day of the event

#============================== reg volume vs return =====================================================#

def reg_vol_ret(data,N_mini,startyear):
    sentratio_and_price_copy = data.copy()
    with pd.option_context('mode.use_inf_as_na', True):
        sentratio_and_price_copy = sentratio_and_price_copy.dropna().reset_index(drop=True)

    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.date < datetime.datetime(startyear, 1, 1) ].index)
    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.N < N_mini].index)

    sentratio_and_price_copy = sentratio_and_price_copy[sentratio_and_price_copy['weekly_return_t+1'].between(
        sentratio_and_price_copy['weekly_return_t+1'].quantile(.02),sentratio_and_price_copy['weekly_return_t+1'].quantile(.98))]  # without outliers

    X = sm.add_constant(sentratio_and_price_copy[['D_N']])  # --- with constant

    y = sentratio_and_price_copy['weekly_return_t+1']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = OLS(yTrain, xTrain)
    res = model.fit()

    xplot = sentratio_and_price_copy['D_N'].values
    yplot = sentratio_and_price_copy['weekly_return_t+1'].values
    plt.scatter(xplot,yplot)
    plt.xlabel('Change in number of messages posted per week')
    plt.ylabel('weekly_return_t+1')
    plt.title('Activity vs Weekly Return')
    plt.grid()
    #plt.savefig('activity-volume.jpg')
    plt.show()
    print(res.summary())

reg_vol_ret(sentratio_and_price,500,2012)


#============================== reg bidask spread vs Bullishness =============================================#

def reg_spread_bullishness(data,N_mini,startyear):
    sentratio_and_price_copy = data.copy()
    with pd.option_context('mode.use_inf_as_na', True):
        sentratio_and_price_copy = sentratio_and_price_copy.dropna().reset_index(drop=True)

    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.date < datetime.datetime(startyear, 1, 1) ].index)
    sentratio_and_price_copy = sentratio_and_price_copy.drop(sentratio_and_price_copy[sentratio_and_price_copy.N < N_mini].index)
    X = sm.add_constant(sentratio_and_price_copy[['Bullishness']])  # --- with constant

    y = sentratio_and_price_copy['spread']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = OLS(yTrain, xTrain)
    res = model.fit()

    xplot = sentratio_and_price_copy['Bullishness'].values
    yplot = sentratio_and_price_copy['spread'].values
    plt.scatter(xplot,yplot)
    plt.xlabel('Bullishness')
    plt.ylabel('Bid-Ask spread')
    plt.title('Bullishness vs Bid-Ask spread')
    plt.grid()
    #plt.savefig('activity-bidask.jpg')
    plt.show()
    print(res.summary())

reg_spread_bullishness(sentratio_and_price,20,2012)   #reg is signif but plot is meh
