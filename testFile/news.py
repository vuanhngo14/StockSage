import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import finnhub
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

api_key = "cn0ah7pr01qkcvkfucv0cn0ah7pr01qkcvkfucvg"; 

import time
max_requests_per_minute = 180


def news_sentiment(end_date, company_code, api_key):

    finnhub_client = finnhub.Client(api_key=api_key)

    # Get all the news of that day for the company
    data = finnhub_client.company_news(company_code, _from = end_date, to=end_date)
    print(data)


    # for i in range(0, len(data)):
    #     headline = (data[i]['headline']) # headline of the news 
    #     print(headline)

    return 0
    
print(news_sentiment('2023-02-06', 'AAPL', api_key))

