from urllib.request import urlopen
import cloudpickle as cp
from preprocessing import preprocess
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@st.experimental_singleton
def load_model(url_model):
    return cp.load(urlopen(url_model))


mod = load_model("https://github.com/marcaureledivernois/tweet_sklearn/releases/download/v1.0/ros_fit.sav")

st.title('Tweets Sentiment Analysis')
st.write('This NLP application is trained on 90 million tweets scraped from the API of Stocktwits.com from 2010 to 2020. Chose "Sentiment classifier" to classify any tweet you write, "Polarity Time-Series" for interactive polarity plots, and "Download data" to get my data.')
option = st.selectbox('What tool do you want to use?',('Sentiment Classifier', 'Polarity Time-Series', 'Download Data'))
#st.subheader('*By Marc-Aurèle Divernois*')

if option == "Sentiment Classifier":
    text = st.text_area(label="Enter message")

    if text:
        clean = preprocess(text)

        bullish_proba = mod.predict_proba([text])[0][1]

        if bullish_proba > 0.6:
            sent = 'Bullish'
        elif 0.4 < bullish_proba < 0.6:
            sent = 'Neutral'
        elif bullish_proba < 0.4:
            sent = 'Bearish'
        else:
            sent = 'error'

        st.write('**Your message :**' , text)
        st.write('**Preprocessed message :**' , clean)
        st.write('**This message is classified as :**', sent)

        feature_names = np.array(mod.named_steps["tfidfvectorizer"].get_feature_names())
        coefs = mod.named_steps["logisticregression"].coef_.flatten()

        feat_importance = np.multiply(mod.named_steps.tfidfvectorizer.transform([text]).toarray(), mod.named_steps.logisticregression.coef_)
        feat_importance = feat_importance[feat_importance != 0 ]

        # Zip coefficients and names together and make a DataFrame
        zipped = zip(feature_names[np.nonzero(mod.named_steps.tfidfvectorizer.transform([text]).toarray()[0])[0]], feat_importance)
        dfz = pd.DataFrame(zipped, columns=["feature", "value"])# Sort the features by the absolute value of their coefficient
        dfz["abs_value"] = dfz["value"].apply(lambda x: abs(x))
        dfz["colors"] = dfz["value"].apply(lambda x: "green" if x > 0 else "red")
        dfz = dfz.sort_values("abs_value", ascending=False)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.barplot(x="feature",
                    y="value",
                    data=dfz.head(20),
                   palette=dfz.head(20)["colors"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
        ax.set_title("Feature importance", fontsize=25)
        ax.set_ylabel("Coef", fontsize=22)
        ax.set_xlabel("Feature Name", fontsize=22)
        st.pyplot(fig)

    
if option == "Polarity Time-Series":
    @experimental_memo
    def load_df(filename):
        return pd.read_csv(filename)
    
    sentratio_and_price = load_df('data/sentratio_and_price.csv')
    eventlist = load_df('data/eventlist.csv')
    def plot_activity(ticker):
        cut_act = sentratio_and_price[(sentratio_and_price['ticker']==ticker) & (sentratio_and_price['businessday']==True)]
        cut_event = eventlist[eventlist['ticker']==ticker]

        fig, ax = plt.plot(cut_act['date'],cut_act['N'],'black')
        plt.scatter(cut_event.loc[cut_event['Type']=='Good','date'],cut_event.loc[cut_event['Type']=='Good','N'],marker='^', s=80, facecolors='none', edgecolors='g')
        plt.scatter(cut_event.loc[cut_event['Type'] == 'Neutral', 'date'], cut_event.loc[cut_event['Type'] == 'Neutral', 'N'],
                    s=80, facecolors='none', edgecolors='gray')
        plt.scatter(cut_event.loc[cut_event['Type'] == 'Bad', 'date'], cut_event.loc[cut_event['Type'] == 'Bad', 'N'],
                    s=80, facecolors='none',marker='v', edgecolors='r')
        plt.xlabel('Date')
        plt.ylabel('N')
        plt.title('Activity - ' + ticker)
        st.pyplot(fig)

    plot_activity('AAPL')
        
if option == "Download Data":
    st.download_button(label="Download Tweets Classifier",data=mod,file_name='classifier.sav')
