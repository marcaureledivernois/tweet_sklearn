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

#sentratio_and_price = pd.read_pickle('dailypolarity3.pkl')
#sentratio_and_price_closetoclose

sentratio_and_price = pd.read_pickle('sentratio_and_price_closetoclose_adj.pkl')
spy = sentratio_and_price[sentratio_and_price['ticker']=='SPY']
sentratio_and_price = sentratio_and_price.drop(sentratio_and_price[sentratio_and_price['ticker']=='SPY'].index)

market_polarity = []
for d in pd.date_range(start='2010-01-01', end='2020-03-23', freq='D').tolist():
    d = pd.to_datetime(d, utc=True)
    cut_tick = sentratio_and_price[sentratio_and_price['date'] == d].copy()
    N_total = np.sum(cut_tick['Nbullish']) + np.sum(cut_tick['Nbullish'])
    cut_tick['weight'] = (cut_tick['Nbullish'] + cut_tick['Nbearish']) / N_total
    cut_tick['Pol'] = (cut_tick['Nbullish'] - cut_tick['Nbearish']) / (cut_tick['Nbullish'] + cut_tick['Nbearish'])
    cut_tick['weighted_pol'] = cut_tick['weight'] * cut_tick['Pol']
    market_pol = np.sum(cut_tick['weighted_pol'])
    market_polarity.append([d, market_pol])



market_polarity = pd.DataFrame.from_records(market_polarity, columns=['date', 'market_polarity'])
market_polarity = pd.merge(market_polarity, spy[['date','Polarity']],  how='left', left_on='date', right_on = 'date')
market_polarity = market_polarity.rename(columns={'Polarity':'spy_polarity'})

market_polarity.corr()

y = market_polarity[['market_polarity']].values
X = market_polarity[['spy_polarity']].values
X = sm.add_constant(X)
par = sm.OLS(y, X, missing='drop').fit().params


plt.scatter(market_polarity['spy_polarity'],market_polarity['market_polarity']) # spy, market
#plt.plot([-0.5, 0.5], [-0.5, 0.5],label = '45° degree line', color = 'black', linewidth = 2)
plt.plot(market_polarity['spy_polarity'], par[1]*market_polarity['spy_polarity'] + par[0], label='y={:.2f}x+{:.2f}'.format(par[0],par[1]) ,color = 'red', linewidth = 2)
plt.xlabel('SPY')
plt.ylabel('Market Polarity')
plt.legend(loc='lower right')
#plt.savefig('ES_cum_ret.jpg')
plt.show()

#market_polarity2
#with open('market_polarity2_closetoclose_adj.pkl', 'wb' ) as f:
#    pickle.dump(market_polarity, f)

nunique = sentratio_and_price[['date','ticker']].groupby('date')['ticker'].nunique()




