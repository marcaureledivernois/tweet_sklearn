## StockTwits Classified Sentiment and Stock Returns (forthcoming in Digital Finance)

Authors : 

* Marc-Aurèle Divernois
* Damir Filipovic

This is the repository for the paper "StockTwits Classified Sentiment and Stock Returns", jointly written with Pr. Damir Filipovic.

This project uses 90 million messages scraped from StockTwits (see repo stocktwits-download). 
We classify thoses messages in three categories using logistic regression of TFIDF vectorized messages.
We create firm-individual sentiment polarity and market-aggregated time-series. 
Polarity is positively associated with contemporaneous stock returns. 
On average, polarity is not able to predict next-day stock returns but when we focus on specific events (defined as sudden peak of message volume), 
polarity has predictive power on abnormal returns.

## You can find our StockTwits data in pickle format at the following Dropbox links. 
https://www.dropbox.com/s/btyvj3r1lato783/df_withcorona_clean_1_with_proba_opti_and_hour.pkl?dl=0
https://www.dropbox.com/s/d63t6hy2ggnolc5/df_withcorona_clean_2_with_proba_opti_and_hour.pkl?dl=0

Please acknowledge our paper using the following citation :

M.-A. Divernois and D. Filipovic. StockTwits Classified Sentiment and Stock Returns. Digital Finance (forthcoming), 2023.

