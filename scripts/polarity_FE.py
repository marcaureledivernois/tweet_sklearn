import pandas as pd
import numpy as np
import os
from linearmodels import PanelOLS
import statsmodels.api as sm

#todo fact: some ppl are more optimistic on some firms than others in an aggregate level. find if we can explain why (cash, revenues, ... , other fixed effects (reputation),..?
sentratio_and_price = pd.read_pickle('sentratio_and_price.pkl')
sentratio_and_price[['ticker','Polarity']].groupby('ticker').mean()

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_new.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_new.pkl')])

def Nbullish(x):
    bull = x['sent_merged'] == 1
    return x.loc[bull, 'sent_merged'].count()

def Nbearish(x):
    bear = x['sent_merged'] == -1
    return x.loc[bear, 'sent_merged'].count()

#=========================== empty dataframe =========================================
from itertools import product

tickers = df['ticker'].unique().tolist()
uniques = [pd.date_range(start='2013-01-01', end='2020-03-23', freq='Q').tolist(), tickers]
cadre = pd.DataFrame(product(*uniques), columns = ['date','ticker'])
cadre = cadre.sort_values(['ticker', 'date'], ascending=[1, 1])

#=========================== populate ================================================

sentratio = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker']).apply(Nbullish))
sentratio.rename(columns={0:'Nbullish'}, inplace=True)
sentratio['Nbearish'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker']).apply(Nbearish))
sentratio['N'] = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['sent_merged'].count())['sent_merged']
sentratio['Polarity'] = (sentratio['Nbullish']-sentratio['Nbearish'])/(10+sentratio['Nbearish']+sentratio['Nbullish']) # I add 10 to the denom so polarity on days with few bullish and 0 bearish are close to 0
sentratio['Agreement'] = 1-np.sqrt(1-((sentratio['Nbullish']-sentratio['Nbearish'])/(10+sentratio['Nbearish']+sentratio['Nbullish']))**2)
sentratio = sentratio.reset_index()

sentratio = pd.merge(cadre, sentratio,  how='left', left_on=['date','ticker'], right_on = ['date','ticker'])

N_byfirm = pd.DataFrame(sentratio.groupby('ticker')['N'].median())
N_byfirm['big'] = N_byfirm['N']>50
tickers_to_keep = list(N_byfirm[N_byfirm['big']].index)

sentratio = sentratio[sentratio['ticker'].isin(tickers_to_keep)]
sentratio = sentratio.reset_index(drop=True)

#=========================== accounting ================================================

matching_table = pd.read_csv(os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'table_de_corr_changetickers.csv'),sep=';')

#correction is done slightly below

# accounting : accounting data
accounting = pd.read_csv("C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\extended_db.csv",sep=';')
accounting['datadate'] = pd.to_datetime(accounting['datadate'],format="%Y%m%d")
accounting['Qdate'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in accounting.datadate]

full_db = pd.merge(sentratio, accounting,  how='left', left_on=['date','ticker'], right_on = ['Qdate','tic'])

full_db = full_db.drop(['indfmt','consol','popsrc','datafmt','curcdq','staltq','costat','tic','datafmt','popsrc','consol'], axis=1)

# create variables
full_db['cash/ta'] = full_db['cheq'] / full_db['atq']
full_db['ni/ta'] = full_db['niy'] / full_db['atq']
full_db['size'] = np.log(full_db['prccq'] * full_db['cshoq'])

full_db = full_db.dropna(subset=['cash/ta','ni/ta','size','Polarity']).copy().reset_index(drop=True)

full_db = full_db.set_index(['ticker','date'])
fe_mod = PanelOLS(full_db.Polarity,sm.add_constant(full_db[['cash/ta' , 'size', 'ni/ta']]), entity_effects=True)
fe_res = fe_mod.fit(cov_type='clustered', cluster_entity=True)
df_cor_res = pd.concat([full_db.Polarity, fe_res.fitted_values],1).dropna()
print(fe_res)
print(df_cor_res.corr().iloc[0,1] ** 2)
