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

#df = pd.read_pickle('df_clean.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_1.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_2.pkl')
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_new.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_new.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl')])  #sent with opti thresholds

#=========================== use only labeled tweets =========================================

#df_lab = df.loc[((df['sent']==-1) | (df['sent']==1))]
#df_lab = df_lab.reset_index(drop=True)

#def bullishness(x):
#    bull = x['sent'] == 1
#    bear = x['sent'] == -1
#    return x.loc[bull, 'sent'].sum() / (x.loc[bull, 'sent'].sum() + np.abs(x.loc[bear, 'sent'].sum()))


#sentratio = pd.DataFrame(df_lab.set_index('date').groupby([pd.Grouper(freq='W'),'ticker']).apply(bullishness))
#sentratio.rename(columns={0:'Bullishness'}, inplace=True)
#sentratio['N'] = pd.DataFrame(df_lab.set_index('date').groupby([pd.Grouper(freq='W'),'ticker'])['sent'].count())['sent']
#sentratio = sentratio.reset_index()


#=========================== use also classified tweets =========================================

def bullishness(x):
    bull = x['sent_merged'] == 1
    bear = x['sent_merged'] == -1
    return x.loc[bull, 'sent_merged'].sum() / (x.loc[bull, 'sent_merged'].sum() + np.abs(x.loc[bear, 'sent_merged'].sum()))

def bullishness_with_neutral(x):
    bull = x['sent_merged'] == 1
    bear = x['sent_merged'] == -1
    neutral = x['sent_merged'] == 0
    return x.loc[bull, 'sent_merged'].sum() / (x.loc[bull, 'sent_merged'].sum() + x.loc[neutral, 'sent_merged'].shape[0]
                                               + np.abs(x.loc[bear, 'sent_merged'].sum()))


sentratio = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W'),'ticker']).apply(bullishness_with_neutral))
sentratio.rename(columns={0:'Bullishness'}, inplace=True)
sentratio['N'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W'),'ticker'])['sent_merged'].count())['sent_merged']
sentratio = sentratio.reset_index()



#============================== merge with accounting data ============================================================#

matching_table = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'table_de_corr_changetickers.csv'),sep=';')

#correction is done slightly below

# accounting : accounting data
accounting = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\extended_db.csv",sep=';')
accounting['datadate'] = pd.to_datetime(accounting['datadate'],format="%Y%m%d")
accounting['Qdate'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in accounting.datadate]

# prices_db : stock price end of month

prices_db = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\monthly_prices.csv",sep=',')
prices_db.tic = prices_db.tic.replace(matching_table.set_index('old').new.to_dict()) #correct tickers with matching table
prices_db['datadate'] = pd.to_datetime(prices_db['datadate'],format="%Y%m%d")


# sentratio_and_price : merge sentratio and stock price -> only stock price at end of each quarter is matched

#sentratio_and_price = pd.merge(sentratio, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])            #using sentratio pre classif
#sentratio_and_price = pd.merge(sentratio_new, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])        #using sentratio post classif
sentratio_and_price = pd.merge(sentratio, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])      #using sentratio_merged
sentratio_and_price['adjprice'] = sentratio_and_price['prccm']/sentratio_and_price['ajexm']

# full_db : sentratio, adjusted stock price from monthly, accounting (with stock price coming from accounting data <-> time match issue)
#adjprice : adjusted price coming from monthly stock price data   <-> problem bcz this is not quarterly return, this is monthly return
#adjprice2 : adjusted price coming from quarterly accounting data <-> time match issue

full_db = pd.merge(sentratio_and_price, accounting,  how='left', left_on=['date','ticker'], right_on = ['Qdate','tic'])
full_db = full_db.sort_values(['ticker', 'date'], ascending=[1, 1])
full_db['adjprice2'] = full_db['prccq']/full_db['adjex']
full_db['return'] = full_db.groupby('ticker').adjprice.pct_change()

full_db = full_db.drop(['gvkey_x', 'iid','indfmt','consol','popsrc','datafmt','curcdq','staltq','costat'], axis=1)
full_db['uinvq'] = full_db['uinvq'].fillna(0)
full_db['dlrsn'] = full_db['dlrsn'].fillna(0)
full_db['dldte'] = full_db['dldte'].fillna(0)

# create variables
full_db['log_cash/ta'] = np.log(full_db['cheq'] / full_db['atq'])
full_db['ni/ta'] = full_db['niy'] / full_db['atq']
full_db['size'] = np.log(full_db['prccq'] * full_db['cshoq'])
full_db['log_bullishness'] = np.log(full_db['Bullishness'])

#================================== group by size ============================================================#

AvgSizeFirms = full_db.groupby(['ticker']).mean()['size']
AvgSizeFirms = AvgSizeFirms.sort_values()
AvgSizeFirms.dropna(inplace=True)
AvgSizeFirms_small = AvgSizeFirms[int(AvgSizeFirms.shape[0]/2):]
AvgSizeFirms_big = AvgSizeFirms[:int(AvgSizeFirms.shape[0]/2)]

#================================== group by size ============================================================#

thresh = -0.3
DD_lag =-1
N_mini = 50

import seaborn as sns

def test_distrib(thresh,DD_lag,N_mini,full_db, subset_tickers):
    full_db_copy = full_db.copy()
    if subset_tickers is not None:
        full_db_copy = full_db_copy[full_db_copy['ticker'].isin(subset_tickers)]
    full_db_copy = full_db_copy.drop(full_db_copy[full_db_copy.N < N_mini].index)

    full_db_copy['drawdown_dummy'] = (full_db_copy['return'] < thresh) * 1

    # set drawdown dummy to 1 for defaulted firms (overwrite)
    defaulted2 = full_db_copy.loc[(full_db_copy['dlrsn'] == 2) | (full_db_copy['dlrsn'] == 3)]
    idx2 = defaulted2.groupby('ticker').tail(1).index.values.astype(int)
    full_db_copy.loc[idx2, 'drawdown_dummy'] = 1

    full_db_copy['drawdown_dummy_lag'] = full_db_copy.groupby(['ticker'])['drawdown_dummy'].shift(DD_lag)

    #drop na
    with pd.option_context('mode.use_inf_as_na', True):
        full_db_copy = full_db_copy.dropna().reset_index(drop=True)
    #full_db_nonan['sdd_log_bullishness'] = (full_db_nonan['log_bullishness']-np.mean(full_db_nonan['log_bullishness']))\
    #                                       /np.std(full_db_nonan['log_bullishness'])
    #full_db_nonan['sdd_log_bearishness'] = (full_db_nonan['log_bearishness']-np.mean(full_db_nonan['log_bearishness']))\
    #                                       /np.std(full_db_nonan['log_bearishness'])


    full_db_copy = full_db_copy[full_db_copy['ni/ta'].between(full_db_copy['ni/ta'].quantile(.02),
                                                                 full_db_copy['ni/ta'].quantile(
                                                                     .98))]  # without outliers


    ax = sns.boxplot(x="drawdown_dummy_lag", y="Bullishness", data=full_db_copy, whis=1.5)
    ax = sns.stripplot(x='drawdown_dummy_lag', y="Bullishness", data=full_db_copy, color="orange", jitter=0.03, size=1.5)
    plt.title('threshold: ' + str(thresh) + ', N_mini ' + str(N_mini))
    plt.ylabel("Bullishness Ratio")
    plt.xlabel("Drawdown dummy")
    #plt.savefig('boxplot_bearish.jpg')
    plt.show()

for thresh in [-0.01,-0.02,-0.03,-0.04,-0.05,-0.1,-0.2,-0.4]:
    for N_mini in [0,5,10,20,30,35,40,50]:
        test_distrib(thresh,DD_lag,N_mini, full_db, None)












































def plot_boxplot(data,var):
    fig, ax = plt.subplots()
    boxes = [
        {
            'label' : "DD = 0",
            'whislo': data[data['drawdown_dummy_lag']==0][var].quantile(0.05),    # Bottom whisker position
            'q1'    : data[data['drawdown_dummy_lag']==0][var].quantile(0.25),    # First quartile (25th percentile)
            'med'   : data[data['drawdown_dummy_lag']==0][var].quantile(0.5),    # Median         (50th percentile)
            'q3'    : data[data['drawdown_dummy_lag']==0][var].quantile(0.75),    # Third quartile (75th percentile)
            'whishi': data[data['drawdown_dummy_lag']==0][var].quantile(0.95),    # Top whisker position
            'fliers': data[(data['drawdown_dummy_lag']==0) &
                           ((data[var] < data[data['drawdown_dummy_lag']==0][var].quantile(0.05)) |
                             (data[var] > data[data['drawdown_dummy_lag']==0][var].quantile(0.95)))][var]       # Outliers
        },
    {
            'label' : "DD = 1",
            'whislo': data[data['drawdown_dummy_lag']==1][var].quantile(0.05),    # Bottom whisker position
            'q1'    : data[data['drawdown_dummy_lag']==1][var].quantile(0.25),    # First quartile (25th percentile)
            'med'   : data[data['drawdown_dummy_lag']==1][var].quantile(0.5),    # Median         (50th percentile)
            'q3'    : data[data['drawdown_dummy_lag']==1][var].quantile(0.75),    # Third quartile (75th percentile)
            'whishi': data[data['drawdown_dummy_lag']==1][var].quantile(0.95),    # Top whisker position
            'fliers': data[(data['drawdown_dummy_lag']==1) &
                           ((data[var] < data[data['drawdown_dummy_lag']==1][var].quantile(0.05)) |
                             (data[var] > data[data['drawdown_dummy_lag']==1][var].quantile(0.95)))][var]        # Outliers
        }
    ]

    ax.bxp(boxes, showfliers=False)
    ax.set_ylabel(var)
    #plt.savefig("boxplot.png")
    plt.title(var)
    plt.grid()
    plt.show()


plot_boxplot(Xout,'log_bullish_proba')
plot_boxplot(Xout,'cash/ta')
plot_boxplot(Xout,'ni/ta')
plot_boxplot(Xout,'size')
