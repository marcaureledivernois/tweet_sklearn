from urllib.request import urlopen
import cloudpickle as cp
from preprocessing import preprocess
import streamlit as st

orig_mod = cp.load(urlopen("https://github.com/marcaureledivernois/tweet_sklearn/releases/download/v1.0/original_fit.sav"))

st.title('StockTwits Sentiment classifier')
text = st.text_area(label="Enter message")

if text:
    clean = preprocess(text)

    bullish_proba = orig_mod.predict_proba([text])[0][0]

    if bullish_proba > 0.72:
        sent = 'Bullish'
    elif 0.5 < bullish_proba < 0.72:
        sent = 'Neutral'
    elif bullish_proba < 0.5:
        sent = 'Bearish'
    else:
        sent = 'error'

    st.write('The score of your message is :', bullish_proba)
    st.write('Given our optimal thresholds, the message is classified as', sent)
