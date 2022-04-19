from urllib.request import urlopen
import cloudpickle as cp
from preprocessing import preprocess
import streamlit as st

orig_mod = cp.load(urlopen("https://github.com/marcaureledivernois/tweet_sklearn/releases/download/v1.0/original_fit.sav"))

text = "Hello my nAme is marc-aurele, i'd love to. would you like to? y'all nice 3."
clean = preprocess(text)

bullish_proba = orig_mod.predict_proba([text])[0][0]

st.write(bullish_proba)
