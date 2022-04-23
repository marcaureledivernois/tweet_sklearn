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

st.title('StockTwits Sentiment classifier')
st.subheader('*By Marc-Aurèle Divernois*')
st.write('Sentiment classifier trained on 90 million messages from Stocktwits.com. Messages are classified into bullish, neutral or bearish classes.')
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

    feature_names[np.nonzero(mod.named_steps.tfidfvectorizer.transform([text]).toarray()[0])[0]]
    coefs[np.nonzero(mod.named_steps.tfidfvectorizer.transform([text]).toarray()[0])[0]]

    feat_importance = np.multiply(mod.named_steps.tfidfvectorizer.transform([text]).toarray(), mod.named_steps.logisticregression.coef_)
    feat_importance = feat_importance[feat_importance != 0 ]

    # Zip coefficients and names together and make a DataFrame
    zipped = zip(feature_names[np.nonzero(mod.named_steps.tfidfvectorizer.transform([text]).toarray()[0])[0]], feat_importance)
    dfz = pd.DataFrame(zipped, columns=["feature", "value"])# Sort the features by the absolute value of their coefficient
    dfz["abs_value"] = dfz["value"].apply(lambda x: abs(x))
    dfz["colors"] = dfz["value"].apply(lambda x: "green" if x > 0 else "red")
    dfz = dfz.sort_values("abs_value", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    sns.barplot(x="feature",
                y="value",
                data=dfz.head(20),
               palette=dfz.head(20)["colors"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
    ax.set_title("Feature importance", fontsize=25)
    ax.set_ylabel("Coef", fontsize=22)
    ax.set_xlabel("Feature Name", fontsize=22)
    st.pyplot(fig)

