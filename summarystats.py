import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hmean
from scipy.stats import norm
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#df = pd.read_pickle('df_clean.pkl')                                                                                     #tweets 2012-2018
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_new.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_new.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl')])  #sent with opti thresholds


#==================================== User-labeled Tweets - Pre classification ==========================================================

aggreg = {'Bullishness': lambda x: (x==1).sum()/(abs(x)).sum(),
          'Bearishness': lambda x: (x==-1).sum()/(abs(x)).sum(),
          'Nbullish': lambda x: (x==1).sum(),
          'Nbearish': lambda x: (x==-1).sum(),
          'Nundefined': lambda x: (x==0).sum()}

ratio_ts = df.set_index('date').groupby([pd.Grouper(freq='D')])['sent'].agg(aggreg).reset_index()
ratio_ts['Undefined_perc'] = ratio_ts['Nundefined']/(ratio_ts['Nundefined']+ratio_ts['Nbullish']+ratio_ts['Nbearish'])
ratio_ts['Bullish_perc'] = ratio_ts['Nbullish']/(ratio_ts['Nundefined']+ratio_ts['Nbullish']+ratio_ts['Nbearish'])
ratio_ts['Bearish_perc'] = ratio_ts['Nbullish']/(ratio_ts['Nundefined']+ratio_ts['Nbullish']+ratio_ts['Nbearish'])


barWidth = 0.85
plt.plot(ratio_ts['date'],ratio_ts['Bullishness'],'r')
plt.plot(ratio_ts['date'],ratio_ts['Undefined_perc'],'b')
plt.legend(('User-labeled 3M Bullishness', 'Unclassified'),
           loc='upper right')
plt.title('User-labeled Tweets - Pre classification')


#==================================== User-labeled Tweets - Post classification ==========================================================

aggreg = {'Bullishness': lambda x: (x==1).sum()/(abs(x)).sum(),
          'Bearishness': lambda x: (x==-1).sum()/(abs(x)).sum(),
          'Nbullish': lambda x: (x==1).sum(),
          'Nbearish': lambda x: (x==-1).sum(),
          'Nneutral': lambda x: (x==0).sum()}

ratio_ts_new = df.set_index('date').groupby([pd.Grouper(freq='M')])['sent_merged'].agg(aggreg).reset_index()
ratio_ts_new['Neutral_perc'] = ratio_ts_new['Nneutral']/(ratio_ts_new['Nneutral']+ratio_ts_new['Nbullish']+ratio_ts_new['Nbearish'])

#samething but on sent_merg (=postclassif if preclassif=0, else preclassif) instead
ratio_ts_merged = df.set_index('date').groupby([pd.Grouper(freq='M')])['sent_merged'].agg(aggreg).reset_index()
ratio_ts_merged['Neutral_perc'] = ratio_ts_merged['Nneutral']/(ratio_ts_merged['Nneutral']+ratio_ts_merged['Nbullish']+ratio_ts_merged['Nbearish'])

barWidth = 0.85
plt.plot(ratio_ts_new['date'][5:],ratio_ts_merged['Bullishness'][5:],'r')    #note : i remove first 5 weeks because too volatile due to lack of data
plt.plot(ratio_ts_new['date'][5:],ratio_ts_merged['Neutral_perc'][5:],'b')
plt.legend(('3M Bullishness', 'Neutral sentiment'),
           loc='center right')
plt.title('User-labeled Tweets - Post classification')
plt.savefig('User-labeled Tweets - Post classification.jpg')
plt.show()

#============================== User-labeled 3M Bullishness - Pre vs Post classification ==========================================================

plt.plot(ratio_ts_new['date'],ratio_ts['Bullishness'],'r')
plt.plot(ratio_ts_new['date'],ratio_ts_new['Bullishness'],'b')
plt.legend(('3M Bullishness - Pre classification', '3M Bullishness - Post classification'),
           loc='upper right')
plt.title('3M Bullishness - Pre vs Post classification')
print('Corr coef : ',np.corrcoef(ratio_ts['Bullishness'][3:],ratio_ts_new['Bullishness'][3:])[0][1])
#Corr coef :  0.8544964768590421

#============================== 3M Bullishness vs S&P Return ==========================================================

#todo : lag

spret_df = pd.read_csv("spreturn_q.csv",sep=';')
spret_df['caldt'] = pd.to_datetime(spret_df['caldt'],format="%Y%m%d")
spret_df['caldt'] = pd.to_datetime(spret_df['caldt'],format="%Y%m%d")
sent_return_merge = pd.merge(spret_df, ratio_ts_new,  how='left', left_on=['caldt'], right_on = ['date'])
sent_return_merge = sent_return_merge.dropna().reset_index(drop=True)
sent_return_merge['spret'] = sent_return_merge['spindx'].pct_change()
sent_return_merge.at[0, 'spret'] = 1057.08/919.32-1

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Bullishness Ratio', color=color)
ax1.plot(sent_return_merge['caldt'], sent_return_merge['Bullishness'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('S&P Return', color=color)  # we already handled the x-label with ax1
ax2.plot(sent_return_merge['caldt'],sent_return_merge['spret'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('S&P500 - Correlation : ' + "{0:.2f}".format(sent_return_merge.corr()['Bullishness']['spret']))
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
fig.savefig('S&P500 - 3M Bullishness vs S&P Return.jpg')
print('Corr coef : ',sent_return_merge.corr()['Bullishness']['spret'])
#Corr coef :  0.36884185018022003

#============================== 3M Bullishness vs Stock Return -- individual firms =====================================
#sometimes people react, sometimes people anticipate. on graph we see both, but need a regression to see if people are correct when they anticipate or not.


prices_db = pd.read_csv("monthly_prices.csv",sep=',')
prices_db.at[prices_db['tic']=='SVNT','tic'] = 'SVNT2'
prices_db.at[prices_db['tic']=='SVNTQ','tic'] = 'SVNT'
#TODO overwrite the dumb compustat terminology for bankrupted firms (ex: SVNTQ -> SVNT) (check logit_drawdown.py table_de_corr)
prices_db['datadate'] = pd.to_datetime(prices_db['datadate'],format="%Y%m%d")
prices_db['adjprice'] = prices_db['prccm']/prices_db['ajexm']

aggreg = {'Bullishness': lambda x: (x==1).sum()/(abs(x)).sum(),
          'Bearishness': lambda x: (x==-1).sum()/(abs(x)).sum(),
          'Nbullish': lambda x: (x==1).sum(),
          'Nbearish': lambda x: (x==-1).sum(),
          'Nundefined': lambda x: (x==0).sum()}

sentratio_new = df.set_index('date').groupby([pd.Grouper(freq='Q'),'ticker'])['sent_new'].agg(aggreg).reset_index()
sent_return_merge_company = pd.merge(sentratio_new, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])
sent_return_merge_company = sent_return_merge_company.sort_values(by=['ticker','date'], ascending=True)
sent_return_merge_company['return'] = sent_return_merge_company.groupby('ticker').adjprice.pct_change()

sentratio_new_monthly = df.set_index('date').groupby([pd.Grouper(freq='M'),'ticker'])['sent_new'].agg(aggreg).reset_index()
sent_return_merge_company_monthly = pd.merge(sentratio_new_monthly, prices_db,  how='left', left_on=['date','ticker'], right_on = ['datadate','tic'])
sent_return_merge_company_monthly = sent_return_merge_company_monthly.sort_values(by=['ticker','date'], ascending=True)
sent_return_merge_company_monthly['return'] = sent_return_merge_company_monthly.groupby('ticker').adjprice.pct_change()


def plot_bullishness_return_company(ticker):
    df = sent_return_merge_company[sent_return_merge_company['tic']==ticker]
    plt.style.use('seaborn-white')
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Bullishness Ratio', color=color)
    #ax1.plot(df['date'], df['Bullishness'].shift(1), color=color)
    ax1.plot(df['date'], df['Bullishness'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Return', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df['return'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    print('Corr coef : ', df.corr()['Bullishness']['return'])
    plt.title(ticker + ' - Correlation : ' + "{0:.2f}".format(df.corr()['Bullishness']['return']))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    #fig.savefig(ticker + ' - 3M Bullishness vs Return.jpg')

plot_bullishness_return_company('SPY')

# svnt = sent_return_merge_company[sent_return_merge_company['tic']=='SVNT']

# svnt : company selling biologic drug. bankruptcy due to overestimation of the market. sept2012 : 300% return because positive opinion from the american comittee to sell outside US
# TODO try on other firms (smaller, etc) -> merge with acc data and select firms based on size for instance, then aggregate. pb : need compute return of pf

#TODO sudden attention of investors (captured by sudden increase in volume of tweets) : does it mean something?   ---> VOLUME OF TWEETS TO PLOT?

#========================== Categories repartition ==========================================================

ax_pre = sns.countplot(x= 'sent',data = df, palette=['green',"lightgray",'red'], order=[1, 0, -1])    # pre classif
ax_pre.set_title('Number of messages user-labeled')
ax_pre.set_ylabel('Count')
ax_pre.set_xlabel('Label')
ax_pre.set_xticks(range(3))
ax_pre.set_xticklabels(['Bullish','Unlabeled','Bearish'])
plt.show()

ax_post = sns.countplot(x= 'sent_new',data = df, palette=['green',"lightgray",'red'], order=[1, 0, -1])    # post classif
ax_post.set_title('Number of messages classified')
ax_post.set_ylabel('Count')
ax_post.set_xlabel('Label')
ax_post.set_xticks(range(3))
ax_post.set_xticklabels(['Bullish','Neutral','Bearish'])
plt.show()

#=================================== stats : Countvectorizer ========================================================

dftrain = df.loc[((df['sent']==-1) | (df['sent']==1)) & (df['clean_text']!='')]
dftrain = dftrain.reset_index(drop=True)

#TODO !! not a todo but a warning : be careful when removing duplicates. i need them for time serie of each firm
dftrain = dftrain.drop_duplicates(subset="id_msg")
dftrain = dftrain.reset_index(drop=True)

cvec = CountVectorizer(max_features=10000 , stop_words='english', ngram_range = (1,3))
cvec.fit(dftrain['clean_text'])

neg_train = dftrain[dftrain.sent == -1]
pos_train = dftrain[dftrain.sent == 1]
neg_doc_matrix = cvec.transform(neg_train.clean_text)
pos_doc_matrix = cvec.transform(pos_train.clean_text)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)

def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df['pos_rate'] = term_freq_df['positive'] * 1./term_freq_df['total']
term_freq_df['neg_rate'] = term_freq_df['negative'] * 1./term_freq_df['total']
term_freq_df['pos_freq_pct'] = term_freq_df['positive'] * 1./term_freq_df['positive'].sum()
term_freq_df['neg_freq_pct'] = term_freq_df['negative'] * 1./term_freq_df['negative'].sum()

# columns explanation :
# 'positive' : occurence of term in positive tweets only
# 'negative' : occurence of term in negative tweets only
# 'total'    : occurence of term in all tweets, regardless of sentiment
# 'pos_rate' : occurence of term in positive tweets divided by total occurence of term
# 'neg_rate' : occurence of term in negative tweets divided by total occurence of term
# 'pos_freq_pct' : occurence of term in positive tweet divided by the total occurences of terms in positive tweets
# 'neg_freq_pct' : occurence of term in negative tweet divided by the total occurences of terms in negative tweets

term_freq_df.to_pickle('term_freq_df.pkl')
#term_freq_df = pd.read_pickle('term_freq_df.pkl')

term_freq_df_sortedpos = term_freq_df.sort_values(by='positive', ascending=False).iloc[:20]
term_freq_df_sortedneg = term_freq_df.sort_values(by='negative', ascending=False).iloc[:20]

term_freq_df_sortednegrate = term_freq_df[term_freq_df['neg_rate']<0.98].sort_values(by='neg_rate', ascending=False).iloc[:100]
term_freq_df_sortedposrate = term_freq_df[term_freq_df['pos_rate']<0.97].sort_values(by='pos_rate', ascending=False).iloc[:100]


#======================================== user analysis ========================================================
#TODO number of users, what they are saying, number of followers, distribution of number of followers etc etc
#todo those who publish often are they saying bullish or bearish things on average?

df_no_dupl = df.reset_index(drop=True)

df_no_dupl = df_no_dupl.drop_duplicates(subset="id_user")
df_no_dupl = df_no_dupl.reset_index(drop=True)


plt.style.use('seaborn-whitegrid')
#df_no_dupl['logideas'] = df_no_dupl['ideas'].apply(lambda x: np.log(max(x,1)))
#axideas = sns.distplot(df_no_dupl['logideas'],kde=False, bins=np.arange(0,14))
axideas = sns.distplot(df_no_dupl[df_no_dupl['ideas']>0]['ideas'],kde=False,bins=np.arange(0,1000000))
axideas.set_title('Number of messages posted per user')
axideas.set_ylabel('Number of users')
axideas.set_xlabel('Number of messages')
axideas.set_xscale('log',basex=10)
axideas.set_yscale('log',basey=10)
plt.show()
fig = axideas.get_figure()
fig.savefig('Tweets.jpg')


#df_no_dupl['logfoll'] = df_no_dupl['foll'].apply(lambda x: np.log(max(x,1)))
axfoll = sns.distplot(df_no_dupl['foll'],kde=False, bins=np.arange(0,100000))
axfoll.set_title('Number of followers per user')
axfoll.set_ylabel('Number of users')
axfoll.set_xlabel('Number of followers')
axfoll.set_xscale('log',basex=10)
axfoll.set_yscale('log',basey=10)
plt.show()
fig = axfoll.get_figure()
fig.savefig('Followers.jpg')

#======================================== cloud word ========================================================

stopwords = set(STOPWORDS)

def show_wordcloud(dict, title,save=False):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=40,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate_from_frequencies(dict)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

    if save:
        wordcloud.to_file(title + '.jpg')


term_freq_df['term'] = term_freq_df.index
#in line below : use term_freq_df_sortedposrate term_freq_df_sortednegrate to take rate<0.98 into account for neg and pos rate
#                term_freq_df for negative and positive occurences
dict = {x[0]:x[1] for x in term_freq_df_sortedposrate[['term','pos_rate']].values}
show_wordcloud(dict = dict, title = 'Positive rate words',save=True)

#======================================== firm analysis ========================================================

#distribution number of tweets against number of firms

ntwitperfirm = df[['ticker','date']].groupby('ticker').count()

axtwitperfirm = sns.distplot((ntwitperfirm['date']),kde=False, bins = (1,10,100,1000,10000,100000,1000000))
axtwitperfirm.set_title('Number of messages per ticker')
axtwitperfirm.set_ylabel('Number of tickers')
axtwitperfirm.set_xlabel('Number of messages')
axtwitperfirm.set_xscale('log',basex=10)
plt.show()
fig = axtwitperfirm.get_figure()
fig.savefig('LogTweetsPerFirm.jpg')

#top 10 most discussed tickers

ntwitperfirm = ntwitperfirm.sort_values(by='date', ascending=False)
ntwitperfirm['ticker'] = ntwitperfirm.index

top10firms = ntwitperfirm.head(30).plot.bar(x='ticker', y='date', rot=90, legend=None)  #column "date" is the count... just a bad name but who cares
top10firms.set_title('Top 30 most discussed tickers')
top10firms.set_ylabel('Number of messages')
top10firms.set_xlabel('Ticker')
plt.show()
fig = top10firms.get_figure()
fig.savefig('top30tickers.jpg')

axtwitperfirm = sns.distplot((ntwitperfirm['date']),kde=False, bins = (1,10,100,1000,10000,100000,1000000))
axtwitperfirm.set_title('Number of messages per ticker')
axtwitperfirm.set_ylabel('Number of tickers')
axtwitperfirm.set_xlabel('Number of messages')
axtwitperfirm.set_xscale('log',basex=10)
plt.show()

values, base = np.histogram(ntwitperfirm['date'], bins=40)
#evaluate the cumulative
cumulative = np.cumsum(values)
# plot the cumulative function
#plot the survival function
plt.plot(base[:-1], len(ntwitperfirm['date'])-cumulative, c='green')
plt.show()

#======================================== wtf is aldox ========================================================

aldox = df[df['clean_text'].str.contains('political posturing friend')== True]

# aldox is for Aldoxorubicin, drug against tumors. associated with pharma tweets and ppl very enthusiastic about it
# ex : aldox be on that slide have great faith all wont be fast but this be truly world change

#======================================== trend in msg posted ========================================================
#todo trend in msg posted : global, aapl, amzn, etc..

N = pd.DataFrame(df.set_index('date').groupby(pd.Grouper(freq='W-WED'))['sent_merged'].count())['sent_merged']
N = N.reset_index()

plt.plot(N['date'],N['sent_merged'])
plt.title('Activity - Global')
plt.xlabel('Date')
plt.ylabel('Number of weekly messages posted')
plt.savefig('Activity - Global.jpg', bbox_inches='tight')
plt.show()


N_perfirm = pd.DataFrame(df.set_index('date').groupby([pd.Grouper(freq='W-WED'),'ticker'])['sent_merged'].count())['sent_merged']
N_perfirm = N_perfirm.reset_index()

def plot_Nfirm(data,ticker):
    plt.plot(data[data['ticker']==ticker]['date'], data[data['ticker']==ticker]['sent_merged'])
    plt.title('Activity - ' + ticker)
    plt.xlabel('Date')
    plt.ylabel('Number of weekly messages posted')
    plt.savefig('Activity - ' + ticker + '.jpg', bbox_inches='tight')
    plt.show()

plot_Nfirm(N_perfirm,"AAPL")
plot_Nfirm(N_perfirm,"FB")
plot_Nfirm(N_perfirm,"AMZN")
plot_Nfirm(N_perfirm,"AMD")
plot_Nfirm(N_perfirm,"TSLA")
