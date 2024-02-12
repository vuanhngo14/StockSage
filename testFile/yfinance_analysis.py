import yfinance as yf


def display_news(symbol):
    code = yf.Ticker(symbol)
    hist = code.history(period="1mo")
    recommendation = code.recommendations_summary

    return recommendation


print(display_news('AAPL'))