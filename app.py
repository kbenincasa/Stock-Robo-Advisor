'''Author: Katie Benincasa
Date: 7/17/24
Purpose: This code communicates with the html pages and provides metrics about stocks in addition to creating predictions through machine learnign that utilizes clustering and the removal of outliers for accurate results'''

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
import io
import base64
#Above are all the libraries my group used and the main new one is Flask which allows communication between the html pages and provides the best results

app = Flask(__name__)#initializes flask
feedback_list = []#creates an empty list to add feedback to

def process_stock_data(ticker, start_date, end_date):

    """
    Downloads and processes stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pandas.Series: Series of closing prices.
    """

    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data.dropna(inplace=True)
    data = data.asfreq('B')

    return data

def remove_outliers(data):

    '''
    This function was created by Atlas 
    Purpose: It removes outliers from the data using the IQR method for more accurate reults
    Returns: It returns the cleaned data for further functions '''

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_cleaned = data[(data >= lower_bound) & (data <= upper_bound)]

    return data_cleaned

@app.route('/')#This function runs whatever is below given its input, and in this case it runs as soon as the site is loaded until a different function acts upon it
def index():
    return render_template('index.html')#runs the first html page for users

@app.route('/submit', methods=['POST'])#Runs after the user has chosen what ticker and metric they would like and submitted it
def submit():

    '''
    Purpose: This function uses the users inputs on the first html site and generates what they chose and then brings them to the next html page for the results
    Returns: the next html page with their results
    The code used to generate the metric the user chose is from Aaryan'''

    ticker = request.form['stocks']
    metric = request.form['metrics']
    start_date = '2010-01-01'
    end_date = '2024-01-01'

    if metric == 'predicted_price':
        return predict_stock(ticker, start_date, end_date)
    else:
        data, annualized_return, annualized_volatility, sharpe_ratio = fetch_stock_data(ticker, start_date, end_date)
        
        if metric == 'annualized_return':
            value = annualized_return
        elif metric == 'annualized_volatility':
            value = annualized_volatility
        elif metric == 'sharpe_ratio':
            value = sharpe_ratio
        else:
            value = "Invalid Metric"

        plot_url = plot_metric(data, metric, annualized_volatility)

        return render_template('results.html', ticker=ticker, metric=metric, value=value, plot_url=plot_url)

def predict_stock(ticker, start_date, end_date):

    '''
    Purpose: This code predicts the stock price using ARIMA and generates the graph for it as well
    Returns: This fuction uses the predicted stock values and MSE to generate the graph and saves it as an image to be sent to the second html page
    The ARIMA code was provided from the original model but altered and the outlier function is creditted to Atlas
    '''

    processed_data = process_stock_data(ticker, start_date, end_date)
    data = processed_data.copy()
    data['Close'] = remove_outliers(data['Close'])
    data.dropna(inplace=True)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    exog_features = ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line']
    train_exog = train_data[exog_features]
    test_exog = test_data[exog_features]
    p = range(2, 5)
    d = range(0, 2)
    q = range(0, 1)
    pdq = list(product(p, d, q))
    best_aic = np.inf
    best_order = None
    best_mdl = None

    for param in pdq:
        try:
            tmp_mdl = ARIMA(train_data['Close'], exog=train_exog, order=param).fit()
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = param
                best_mdl = tmp_mdl
        except:
            continue

    predictions = best_mdl.forecast(steps=len(test_data), exog=test_exog)
    predictions.index = test_data.index
    mse = mean_squared_error(test_data['Close'], predictions)

    plt.figure(figsize=(14, 5))
    plt.plot(train_data.index, train_data['Close'], label='Train')
    plt.plot(test_data.index, test_data['Close'], label='Test')
    plt.plot(predictions.index, predictions, label='Predicted')
    plt.title(f'Stock Price Prediction of {ticker} using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('results.html', plot_url=plot_url, mse=mse, ticker=ticker, metric='Prediction', value='')

def plot_metric(data, metric, annualized_volatility):

    '''
    Purpose: This function is plotting the metric chosen by the user
    Results: returns the graph of the stock metric as an image to go to the html results page
    '''

    plt.figure(figsize=(14, 5))
    if metric == 'annualized_return':
        sns.histplot(data['Daily Return'].dropna() * 252, kde=True, color='green')
        plt.title('Annualized Return Distribution')
    elif metric == 'annualized_volatility':
        sns.histplot(data['Daily Return'].dropna() * np.sqrt(252), kde=True, color='blue')
        plt.title('Annualized Volatility Distribution')
    elif metric == 'sharpe_ratio':
        risk_free_rate = 0.02
        sharpe_ratios = (data['Daily Return'].dropna() * 252 - risk_free_rate) / (annualized_volatility)
        sns.histplot(sharpe_ratios, kde=True, color='red')
        plt.title('Sharpe Ratio Distribution')

    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

@app.route('/feedback', methods=['POST'])#Prompted by choosing one of 10 feedback buttons
def receive_feedback():

    '''
    Purpose: appends the feedback to a list for future reference and thanks the user for feedback
    Results: users feedback in a list and a thank you message
    This code is from Krishna
    '''

    data = request.get_json()
    rating = data.get('rating')
    feedback_list.append(rating)

    return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})

@app.route('/feedback_list')#happens after value is appended to feedback list
def get_feedback_list():
    return jsonify(feedback_list)#returns the list to be accessed on seperate site

def fetch_stock_data(ticker, start_date, end_date):

    """
    Purpose: This code fetches the historical data and calculates the daily returns, annualized return, and annualized volatility 
    Results: It returns all of the necessary data for future use in further functions
    This code is from Aaryan
    """

    risk_free_rate = 0.02
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    annualized_return = data['Daily Return'].mean() * 252
    annualized_volatility = data['Daily Return'].std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    return data, annualized_return, annualized_volatility, sharpe_ratio

if __name__ == '__main__':
    app.run(debug=True)
