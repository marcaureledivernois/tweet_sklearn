from urllib.request import urlopen
import cloudpickle as cp
from preprocessing import preprocess
import streamlit as st


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
