import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#todo https://stackoverflow.com/questions/29547522/python-pandas-to-pickle-cannot-pickle-large-dataframes
#============================== aggregate sentiment ====================================================================

# df = pd.read_pickle('df_clean.pkl')
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl')])  #sent with opti thresholds

#aggreg = {'Bullishness' : lambda x: (x==1).sum()/(abs(x)).sum(),
#          'Bearishness': lambda x: (x==-1).sum()/(abs(x)).sum(),
#          'Nbullish': lambda x: (x==1).sum(),
#          'Nbearish': lambda x: (x==-1).sum(),
#          'Nundefined': lambda x: (x==0).sum()}


#sentratio = df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['sent'].agg(aggreg).reset_index()
#sentratio_new = df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['sent_new'].agg(aggreg).reset_index()
#sentratio_merged = df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['sent_merged'].agg(aggreg).reset_index()

#sentratio_merged['Bullishness_proba'] = df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['bullish_proba'].mean()

sentratio_merged = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['bullish_proba'].mean().reset_index())
sentratio_merged['bearish_proba'] = 1-sentratio_merged['bullish_proba']
sentratio_merged['N'] = df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['bullish_proba'].agg(total='count').reset_index()['total']

#date in df is the max date. for instance 2010-03-31 00:00:00 corresponds to Q1


#============================== correct tickers with matching table ====================================================
# need to do this step because of compustat convention. In compustat, bankrupted firms have their ticker changed and usually
# they add a Q at the end of the ticker. These new tickers will not be matched with those of stocktwits so I need to correct
# them to match those of the stocktwits database. The matching table has been done manually.
# note : the tickers need to be changed in both monthly_prices and in extended_db ( for extended_db I did it with a vlookup
# in excel... but need python for monthly_prices because too big to open in excel)
# ! be careful if I redownload the extended_db, need to redo corretion)

matching_table = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'table_de_corr_changetickers.csv'),sep=';')

#correction is done slightly below

#============================== merge with accounting data ============================================================#

# accounting : accounting data
accounting = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\extended_db.csv",sep=';')
accounting['datadate'] = pd.to_datetime(accounting['datadate'],format="%Y%m%d")
accounting['Qdate'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in  accounting.datadate]

# prices_db : stock price end of month

prices_db = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\monthly_prices.csv",sep=',')
prices_db.tic = prices_db.tic.replace(matching_table.set_index('old').new.to_dict()) #correct tickers with matching table
prices_db['datadate'] = pd.to_datetime(prices_db['datadate'],format="%Y%m%d")

# sentratio_and_price : merge sentratio and stock price -> only stock price at end of each quarter is matched

#sentratio_and_price = pd.merge(sentratio, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])            #using sentratio pre classif
#sentratio_and_price = pd.merge(sentratio_new, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])        #using sentratio post classif
sentratio_and_price = pd.merge(sentratio_merged, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])      #using sentratio_merged
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

#full_db.to_pickle('full_db.pkl')       #corresponds to db tweents 2012-2018 with sent_new (not sent_merged) - which works nicely with all results
#full_db = pd.read_pickle('full_db.pkl')
#===================================== create variables and drawdown dummy =============================================
#todo : index return and adapt drawdown dummy (damir idea). adapt drawdown_dummy to take into account drop of global economy
#todo : maybe better to work with change of sentiment (diff ratio) instead of level of ratio
#todo : keep only firms with at least 10 tweets on a quarter OR weight observations based on number of tweets
#todo : rolling window?
#todo : check if sentiment predicts return or of return predicts sentiment (Regression continuity design?)


full_db = pd.read_pickle('full_db.pkl')

full_db = full_db.reset_index(drop=True)
full_db = full_db.drop(full_db[full_db.N < 30].index)

# ===================================== regress =========================================================================

def regress(thresh, IndepVars,DD_lag = -1, addconstant = True, up_or_down = ''):
    if up_or_down == 'Drawup':
        full_db['drawdown_dummy'] = (full_db['return'] > thresh)*1
    elif up_or_down == 'Drawdown':
        full_db['drawdown_dummy'] = (full_db['return'] < thresh) * 1
    elif up_or_down == 'Combined':
        full_db['drawdown_dummy'] = ((full_db['return'] < - thresh) | (full_db['return'] > thresh)) * 1
    else:
        return print('error in up_or_down')

    #set drawdown dummy to 1 for defaulted firms (overwrite)
    defaulted2 = full_db.loc[(full_db['dlrsn']==2) | (full_db['dlrsn'] == 3)]
    idx2 = defaulted2.groupby('ticker').tail(1).index.values.astype(int)
    full_db.loc[idx2,'drawdown_dummy'] = 1

    full_db['drawdown_dummy_lag'] = full_db.groupby(['ticker'])['drawdown_dummy'].shift(DD_lag)

    #create variables
    full_db['log_cash/ta'] = np.log(full_db['cheq']/full_db['atq'])
    full_db['ni/ta'] = full_db['niy']/full_db['atq']
    full_db['size'] = np.log(full_db['prccq']*full_db['cshoq'])
    #full_db['log_bullishness'] = np.log(full_db['Bullishness'])
    #full_db['log_bearishness'] = np.log(full_db['Bearishness'])
    full_db['log_bullish_proba'] = np.log(full_db['bullish_proba'])
    full_db['log_bearish_proba'] = np.log(full_db['bearish_proba'])

    #drop na
    with pd.option_context('mode.use_inf_as_na', True):
        full_db_nonan = full_db.dropna().reset_index(drop=True)
    #full_db_nonan['sdd_log_bullishness'] = (full_db_nonan['log_bullishness']-np.mean(full_db_nonan['log_bullishness']))\
    #                                       /np.std(full_db_nonan['log_bullishness'])
    #full_db_nonan['sdd_log_bearishness'] = (full_db_nonan['log_bearishness']-np.mean(full_db_nonan['log_bearishness']))\
    #                                       /np.std(full_db_nonan['log_bearishness'])
    full_db_nonan['sdd_log_bullish_proba'] = (full_db_nonan['log_bullish_proba']-np.mean(full_db_nonan['log_bullish_proba']))\
                                           /np.std(full_db_nonan['log_bullish_proba'])
    full_db_nonan['sdd_log_bearish_proba'] = (full_db_nonan['log_bearish_proba']-np.mean(full_db_nonan['log_bearish_proba']))\
                                           /np.std(full_db_nonan['log_bearish_proba'])


    full_db_nonan = full_db_nonan[full_db_nonan['ni/ta'].between(full_db_nonan['ni/ta'].quantile(.02),
                                                                 full_db_nonan['ni/ta'].quantile(
                                                                     .98))]  # without outliers

    if addconstant == False:
        X = full_db_nonan[IndepVars]                     #--- without constant
    else:
        X = sm.add_constant(full_db_nonan[IndepVars])                                       #--- with constant

    y = full_db_nonan['drawdown_dummy_lag']

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    logit_model=sm.Logit(yTrain,xTrain)
    result=logit_model.fit()
    predictions = result.predict(xTest)
    print(result.summary2())
    return result, predictions,X,y, xTrain, xTest, yTrain, yTest


nicenames =  {'sdd_log_bullishness': 'LogBullishness', 'log_cash/ta': 'Log(Cash/TA)', 'ni/ta': 'NI/TA', 'size': 'Size',
              'sdd_log_bearishness' : 'LogBearishness', 'bullish_proba' : '3MS','bearish_proba' : '1-3MS',
              'log_bullish_proba' : 'Log(3MS)', 'log_bearish_proba':'Log(1-3MS)', 'sdd_log_bullish_proba' : 'Log(3MS) sdd.',
              'sdd_log_bearish_proba' : 'Log(1-3MS) sdd.'}

jmptype = 'Combined'

if jmptype == 'Drawdown':
    thresholds = [-0.7,-0.65,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1]  #to use for drawdown
elif (jmptype == 'Drawup')  | (jmptype == 'Combined') :
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9]                             #to use for drawup

for vartoplot in ['log_bullish_proba','log_bearish_proba']:
    params = []
    ci = []
    for thr in thresholds:                                                      #np.linspace(-1,-0.1,91)   #[-1,-0.9,-0.82,-0.71,-0.7,-0.65,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1]
        vars = ['log_bullish_proba', 'log_bearish_proba', 'log_cash/ta', 'ni/ta', 'size']                                 # ['Bearishness', 'cash/ta', 'ni/ta', 'size']
        res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = thr, IndepVars= vars,DD_lag= -1 , addconstant = True, up_or_down = jmptype)
        params.append(res.params[vartoplot])
        ci.append((res.conf_int()[0][vartoplot]-res.conf_int()[1][vartoplot])/2)

    plt.style.use('seaborn-whitegrid')
    plt.plot(thresholds, params, '-', color='gray')
    plt.fill_between(thresholds, np.array(params) - np.array(ci),
                     np.array(params) + np.array(ci), color='gray', alpha=0.2)
    plt.title('Logit Regression - ' + nicenames[vartoplot])
    plt.xlabel(jmptype +  ' Threshold')
    plt.ylabel(nicenames[vartoplot])
    #plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
    plt.show()

#============================  SCATTER PLOT PROBABILITIES  =============================================================

res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = -0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Drawdown')
plt.scatter(pred.index, pred, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Drawdown with Bearishness')
plt.ylabel('Drawdown Probability')
plt.xlabel('index')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('Scatter_Drawdown_Bearishness.jpg')
plt.show()
pred_yout = pd.concat([yout, pred], axis=1)
pred_df = pred_yout.groupby('drawdown_dummy_lag').describe().reset_index()
print(pred_df.to_latex(index=False, float_format="{:0.3f}".format))

res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = -0.3, IndepVars= ['sdd_log_bullishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Drawdown')
plt.scatter(pred.index, pred, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Drawdown with Bullishness')
plt.ylabel('Drawdown Probability')
plt.xlabel('index')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('Scatter_Drawdown_Bullishness.jpg')
plt.show()
pred_yout = pd.concat([yout, pred], axis=1)
pred_df = pred_yout.groupby('drawdown_dummy_lag').describe().reset_index()
print(pred_df.to_latex(index=False, float_format="{:0.3f}".format))


res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = 0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Drawup')
plt.scatter(pred.index, pred, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Drawup with Bearishness')
plt.ylabel('Drawup Probability')
plt.xlabel('index')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('Scatter_Drawup_Bearishness.jpg')
plt.show()
pred_yout = pd.concat([yout, pred], axis=1)
pred_df = pred_yout.groupby('drawdown_dummy_lag').describe().reset_index()
print(pred_df.to_latex(index=False, float_format="{:0.3f}".format))

res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = 0.3, IndepVars= ['sdd_log_bullishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Drawup')
plt.scatter(pred.index, pred, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Drawup with Bullishness')
plt.ylabel('Drawup Probability')
plt.xlabel('index')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('Scatter_Drawup_Bullishness.jpg')
plt.show()
pred_yout = pd.concat([yout, pred], axis=1)
pred_df = pred_yout.groupby('drawdown_dummy_lag').describe().reset_index()
print(pred_df.to_latex(index=False, float_format="{:0.3f}".format))


res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = 0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Combined')
plt.scatter(pred.index, pred, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Combined with Bearishness')
plt.ylabel('Combined Probability')
plt.xlabel('index')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('Scatter_Combined_Bearishness.jpg')
plt.show()
pred_yout = pd.concat([yout, pred], axis=1)
pred_df = pred_yout.groupby('drawdown_dummy_lag').describe().reset_index()
print(pred_df.to_latex(index=False, float_format="{:0.3f}".format))


res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = 0.3, IndepVars= ['sdd_log_bullishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Combined')
plt.scatter(pred.index, pred, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Combined with Bullishness')
plt.ylabel('Combined Probability')
plt.xlabel('index')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('Scatter_Combined_Bullishness.jpg')
plt.show()
pred_yout = pd.concat([yout, pred], axis=1)
pred_df = pred_yout.groupby('drawdown_dummy_lag').describe().reset_index()
print(pred_df.to_latex(index=False, float_format="{:0.3f}".format))

#============================  STRIPS dummy - variable =================================================================

res,pred,Xout,yout, xTrain, xTest, yTrain, yTest= regress(thresh = -0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Drawdown')
plt.scatter(yout, Xout['sdd_log_bearishness'], alpha=0.5, s = 0.1, c = yout, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Scatter plot : Bearishness vs Drawdown')
plt.ylabel('Bearishness')
plt.xlabel('Drawdown dummy')
cb = plt.colorbar()
loc = np.arange(0,max(yout),max(yout)/float(2))
cb.set_ticks(loc)
cb.set_ticklabels(['0','1'])
#plt.savefig('strips_drawdown_bearishness.jpg')
plt.show()

#===================================  boxplot ==========================================================================
import seaborn as sns


res,pred,Xout,yout, xTrain, xTest, yTrain, yTest= regress(thresh = -0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                                                            , addconstant = True, up_or_down = 'Drawdown')
Xout = Xout.join(yout)

ax = sns.boxplot(x="drawdown_dummy_lag", y="log_bullish_proba", data=Xout, whis=1.5)
ax = sns.stripplot(x='drawdown_dummy_lag', y='log_bullish_proba', data=Xout, color="orange", jitter=0.03, size=1.5)
plt.title("log(s(x(m))) distribution")
plt.ylabel("log Bullishness proba")
plt.xlabel("Drawdown dummy")
#plt.savefig('boxplot_bearish.jpg')
plt.show()

ax = sns.boxplot(x="drawdown_dummy_lag", y="log_cash/ta", data=Xout)
ax = sns.stripplot(x='drawdown_dummy_lag', y='log_cash/ta', data=Xout, color="orange", jitter=0.03, size=1.5)
plt.title("Cash/TA distribution")
plt.ylabel("Cash/TA")
plt.xlabel("Drawdown dummy")
plt.savefig('boxplot_cash.jpg')
plt.show()

ax = sns.boxplot(x="drawdown_dummy_lag", y="ni/ta", data=Xout)
ax = sns.stripplot(x='drawdown_dummy_lag', y='ni/ta', data=Xout, color="orange", jitter=0.03, size=1.5)
plt.title("NI/TA distribution")
plt.ylabel("NI/TA")
plt.xlabel("Drawdown dummy")
plt.savefig('boxplot_ni.jpg')
plt.show()

ax = sns.boxplot(x="drawdown_dummy_lag", y="size", data=Xout)
ax = sns.stripplot(x='drawdown_dummy_lag', y='size', data=Xout, color="orange", jitter=0.03, size=1.5)
plt.title("Size distribution")
plt.ylabel("Size")
plt.xlabel("Drawdown dummy")
plt.savefig('boxplot_size.jpg')
plt.show()



#=================================  HISTROGRAM  ========================================================================

bins = np.linspace(0,200,num=40)
plt.hist(Xout[Xout['drawdown_dummy_lag']==1]['log_bullish_proba'],bins,alpha=0.5,label='1',normed=True)
plt.hist(Xout[Xout['drawdown_dummy_lag']==0]['log_bullish_proba'],bins,alpha =0.5,label ='0', normed=True)
plt.legend(loc ='upper left')
plt.show()

#=================================  GROUPBY MEAN  ======================================================================

pd.set_option('display.max_columns', None)
full_db_nonan[['drawdown_dummy_lag','sdd_log_bearishness']].groupby('drawdown_dummy_lag').describe()

#=================================  PRECISION-RECALL  ==================================================================

def perf_measures(yout,pred,thresh=0.5):
    measures = dict()
    measures['TP'] = 1*(yout * (pred>thresh))
    measures['FP'] = 1*((1-yout) * (pred>thresh ))
    measures['FN'] = 1*(yout *  (1 - (pred>thresh)))
    measures['TN'] = 1*((1-yout) * (1 - (pred>thresh)) )
    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() +measures['FN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    print('precision : ' , measures['precision'] , ', recall : ' , measures['recall'], ' , specificity : ' , measures['specificity'])
    return measures

def plot_perf(yout, pred):
    thresholds = np.linspace(0,1,101)
    precisions = []
    recalls = []
    specificities = []

    for thr in thresholds:
        precisions.append(perf_measures(yout, pred, thresh=thr)['precision'])
        recalls.append(perf_measures(yout, pred, thresh=thr)['recall'])
        specificities.append(perf_measures(yout, pred, thresh=thr)['specificity'])

    plt.style.use('seaborn-whitegrid')
    plt.plot(thresholds, precisions, '-', color='red')
    plt.plot(thresholds, recalls, '-', color='blue')
    plt.plot(thresholds, specificities, '-', color='green')

    plt.title('Performance measures')
    plt.xlabel('Threshold')
    plt.ylabel('Performance measures')
    plt.legend(('Precision','Recall','Specificity'), loc='center right')
    #plt.savefig('')
    plt.show()


res,pred,Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = -0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                             , addconstant = True, up_or_down = 'Drawdown')
perf_measures(yout, pred, thresh = 0.03)
plot_perf(yout, pred)

#====================== precision recall curve with sklearn ============================================================
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

res,pred, Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = -0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                                                    , addconstant = True, up_or_down = 'Drawdown')

precis, recal, thresho = precision_recall_curve(yTest,res.predict(xTest))

no_skill = len(yTest[yTest==1]) / len(yTest)

plt.plot([0, 1], [no_skill, no_skill],color='navy', linestyle='--', label='No Skill')
plt.plot(recal, precis, '-', color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Drawdown Prediction')
plt.legend(('No Skill','Logit'))
plt.savefig('pre_rec_curves_drawdown.png')
plt.show()

fpr, tpr, _ = roc_curve(yTest,res.predict(xTest))
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,  '-', color='darkorange', label='Logit (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', label='No Skill', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Drawdown')
plt.legend(loc="lower right")
plt.savefig('roc_drawdown.png')
plt.show()


res,pred, Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = 0.3, IndepVars= ['sdd_log_bullishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                                                    , addconstant = True, up_or_down = 'Drawup')

precis, recal, thresho = precision_recall_curve(yTest,res.predict(xTest))

no_skill = len(yTest[yTest==1]) / len(yTest)

plt.plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--', label='No Skill')
plt.plot(recal, precis, '-', color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Drawup Prediction')
plt.legend(('No Skill','Logit'))
plt.savefig('pre_rec_curves_drawup.png')
plt.show()

fpr, tpr, _ = roc_curve(yTest,res.predict(xTest))
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, '-', color='darkorange', label='Logit (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', label='No Skill', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Drawup')
plt.legend(loc="lower right")
plt.savefig('roc_drawup.png')
plt.show()


res,pred, Xout,yout, xTrain, xTest, yTrain, yTest = regress(thresh = 0.3, IndepVars= ['sdd_log_bearishness', 'log_cash/ta', 'ni/ta', 'size']   ,DD_lag= -1
                                                    , addconstant = True, up_or_down = 'Combined')

precis, recal, thresho = precision_recall_curve(yTest,res.predict(xTest))

no_skill = len(yTest[yTest==1]) / len(yTest)

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy', label='No Skill')
plt.plot(recal, precis, '-', color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Combined Prediction')
plt.legend(('No Skill','Logit'))
plt.savefig('pre_rec_curves_combined.png')
plt.show()

fpr, tpr, _ = roc_curve(yTest,res.predict(xTest))
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,  '-', color='darkorange', label='Logit (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', label='No Skill', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Combined')
plt.legend(loc="lower right")
plt.savefig('roc_combined.png')
plt.show()


#============================  VOLUME, BIDASK,...  =====================================================================
#todo : regression on volume, bidask spread etc



#============================ CORONAVIRUS ==============================================================================
#todo : was corona drawdown predictable?



#============================ TRADING STRATEGY =========================================================================
#todo design strategy such as weight = eps * EqualWeighting + (1-eps) * (SentimentWeighting and cash) and see if trading strategy using senitment induces lower drawdown/risk in potfolio value



