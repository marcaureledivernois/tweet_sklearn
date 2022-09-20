import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout


accounts = pd.read_csv('https://raw.githubusercontent.com/TreasuryHUB-ai/challenge-mad/master/accounts.csv?token=AS7SO7EJFSUZHLO65BVB54TAWZIW6', index_col=0)
balances = pd.read_csv('https://raw.githubusercontent.com/TreasuryHUB-ai/challenge-mad/master/balances.csv?token=AS7SO7DLVUCGQR6VTI7S6QDAWZITS', index_col=0)

balances = balances.reset_index()
balances = pd.merge(balances, accounts[['Account Description','Currency']],  how='left', left_on='AccountID', right_on = 'Account Description')
def stripped(txt):
    return txt.strip()
balances['Amount'] = balances['Amount'].apply(stripped)
balances['Amount'] = balances['Amount'].replace('-','0')
balances['Amount'] = pd.to_numeric(balances['Amount'].str.replace(',', ''))

balances = balances.rename(columns={'Date dd/mm/yyyy': 'date'})
balances['date'] = pd.to_datetime(balances['date'],format="%m/%d/%Y")

curr_to_keep = ['CAD', 'EUR', 'GBP', 'USD']
balances = balances[balances['Currency'].isin(curr_to_keep)]
balances = balances.reset_index(drop=True)

balances_grouped = balances.groupby(['date','Currency']).sum().reset_index()

balances_r = balances_grouped.pivot(index='date', columns = 'Currency', values = 'Amount').reset_index().sort_values('date')

def plot_curr(curr):
    plt.plot(balances_r['date'],balances_r[curr])
    plt.xticks(rotation = 90)
    plt.title(curr)
    plt.show()

plot_curr('USD')
plot_curr('GBP')
plot_curr('EUR')
plot_curr('CAD')



CAD = np.array(balances_r['CAD']).reshape(-1,1)

scaler = StandardScaler()
training_set_scaled = scaler.fit_transform(CAD)

x_train = []
y_train = []
n_future = 4 # next 4 days temperature forecast
n_past = 30 # Past 30 days
for i in range(0,len(training_set_scaled)-n_past-n_future+1):
    x_train.append(training_set_scaled[i : i + n_past , 0])
    y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])

x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1))
