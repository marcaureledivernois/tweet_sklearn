from urllib.request import urlopen
import cloudpickle as cp
from preprocessing import preprocess
import streamlit

orig_mod = cp.load(urlopen("https://github.com/marcaureledivernois/tweet_sklearn/releases/download/v1.0/original_fit.sav"))

text = "Hello my nAme is marc-aurele"
clean = preprocess(text)

bullish_proba = orig_mod.predict_proba([text])[0][0]

