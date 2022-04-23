
## Stocktwits Sentiment Classification App : 

https://share.streamlit.io/marcaureledivernois/tweet_sklearn/main


## StockTwits Classified Sentiment and Stock Returns

Authors : 

* Marc-Aurèle Divernois
* Damir Filipovic

This is the repository for my paper "StockTwits Classified Sentiment and Stock Returns", written jointly with Pr. Damir Filipovic.

This project uses 90 million messages scraped from StockTwits (see repo stocktwits-download). 
We classify thoses messages in three categories using logistic regression of TFIDF vectorized messages.
We create firm-individual sentiment polarity and market-aggregated time-series. 
Polarity is positively associated with contemporaneous stock returns. 
On average, polarity is not able to predict next-day stock returns but when we focus on specific events (defined as sudden peak of message volume), 
polarity has predictive power on abnormal returns.

