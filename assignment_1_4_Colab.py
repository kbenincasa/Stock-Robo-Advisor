# -*- coding: utf-8 -*-
"""Assignment 1.4

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LFhmZvUOa_ZWGAEkDPIkKJfXGzdFbz_N
"""

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from itertools import product

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

    #feature engineering: add moving averages and other indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data.dropna(inplace=True)

    return data

def arima_model(closing_prices):

    """
    Fits an ARIMA model to the closing prices and makes predictions.

    Parameters:
    closing_prices (pandas.Series): Series of closing prices.

    Returns:
    pandas.Series: Predicted values.
    """

    model = ARIMA(closing_prices, order=(5,1,0))
    model_fit = model.fit()
    predictions = model_fit.predict(typ='levels')
    return predictions

def plot_data(actual, predicted, title):

    """
    Plots actual vs predicted stock prices.

    Parameters:
    actual (pandas.Series): Actual closing prices.
    predicted (pandas.Series): Predicted closing prices.
    title (str): Title of the plot.
    """

    plt.figure(figsize=(10,4))
    plt.plot(actual, label='Original')
    plt.plot(predicted, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_mape(actual, predicted):

    """
    Computes the Mean Absolute Percentage Error (MAPE).

    Parameters:
    actual (pandas.Series): Actual values.
    predicted (pandas.Series): Predicted values.

    Returns:
    float: MAPE value.
    """

    mape = mean_absolute_error(actual, predicted)/len(actual) * 100
    return round(mape, 2)

# Start and end dates
start_date='2019-01-01'
end_date='2023-07-21'

# Process stock data
closing_prices_apple = process_stock_data('AAPL', start_date, end_date)
closing_prices_tesla = process_stock_data('TSLA', start_date, end_date)

# Fit ARIMA and make predictions
predictions_apple = arima_model(closing_prices_apple)
predictions_tesla = arima_model(closing_prices_tesla)

# Compute MAPE
mape_apple = compute_mape(closing_prices_apple, predictions_apple)
mape_tesla = compute_mape(closing_prices_tesla, predictions_tesla)

# Print MAPE
print(f"MAPE for Apple: {mape_apple}%")
print(f"MAPE for Tesla: {mape_tesla}%")

# Plot the data
plot_data(closing_prices_apple, predictions_apple, 'Apple Stock Price: Original vs. Predicted')
plot_data(closing_prices_tesla, predictions_tesla, 'Tesla Stock Price: Original vs. Predicted')

"""Data Exploration:
- Plot the distribution of different stock metrics such as annualized return, annualized volatility, and Sharpe ratio using appropriate visualization techniques. Use libraries like matplotlib and seaborn to visualize these distributions. Histograms can be especially helpful.
- Analyze correlations between different stock metrics using the .corr() function provided by pandas and visualize them using a heatmap. Explain your findings. Should any other variables be added based on the entire dataset?
"""

risk_free_rate = 0.02
def fetch_stock_data(ticker, start_date='2010-01-01', end_date='2024-01-01'):
  # Fetch historical data
  data = yf.download(ticker, start=start_date, end=end_date)
  # Calculate daily returns
  data['Daily Return'] = data['Adj Close'].pct_change()
  # Calculate annualized return
  annualized_return = data['Daily Return'].mean() * 252
  # Calculate annualized volatility
  annualized_volatility = data['Daily Return'].std() * np.sqrt(252)
  # Calculate Sharpe ratio
  sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
  return data, annualized_return, annualized_volatility, sharpe_ratio

ticker = 'AAPL'
data, annualized_return, annualized_volatility, sharpe_ratio = fetch_stock_data(ticker)

fig, axes = plt.subplots(3, 1, figsize=(10,15))
# Plot the daily return distribution
sns.histplot(data['Daily Return'].dropna(), kde=True, ax=axes[0], color='blue')
axes[0].set_title(f'Daily Return Distribution for {ticker}')
axes[0].set_xlabel('Daily Return')
axes[0].set_ylabel('Frequency')

# Plot the annualized return distribution
sns.histplot(data['Daily Return'].dropna() * 252, kde=True, ax=axes[1], color='green')
axes[1].set_title(f'Annualized Return Distribution for {ticker}')
axes[1].set_xlabel('Annualized Return')
axes[1].set_ylabel('Frequency')

# Plot the Sharpe ratio distribution
sharpe_ratios = (data['Daily Return'].dropna() * 252 - risk_free_rate) / (data['Daily Return'].dropna().std() * np.sqrt(252))
sns.histplot(sharpe_ratios, kde=True, ax=axes[2], color='red')
axes[2].set_title(f'Sharpe Ratio Distribution for {ticker}')
axes[2].set_xlabel('Sharpe Ratio')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
#Help used by ChatGpt

"""The code above calculates the daily returns, annualized return, annualized volatility, and sharpe ratio of the stock. The funstion uses the ticker symbol, 'AAPL' and then creates three histograms which are, the distribution of daily returns, annualized returns, and sharpe ratios."""

np.random.seed(42)
data = {'annualized_return': np.random.normal(0.1, 0.15, 1000), 'annualized_volatility': np.random.normal(0.2, 0.05, 1000), 'sharpe_ratio': np.random.normal(0.5, 0.2, 1000)}
# Convert to DataFrame
df = pd.DataFrame(data)
# Plot the heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Stock Metrics')
plt.show()
#Help used by ChatGpt

"""This code above makes a a dataset with annualized returns, annualized volatility, and sharpe ratios and visualizes in using a heat map.

Model Enhancement:
- Implement a method to handle outliers in the dataset before feeding it into the clustering model. Use techniques like Box-Plots or the Interquartile Range (IQR) method. Explain your approach and how it helps in better model training.
- Use different ARIMA prediction algorithms or a different algorithm like Neural Networks to predict stock market and use the most suitable model for prediction. Explain your reasoning.
"""

#box plot to visualize the data
def plot_box(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data)
    plt.title('Box Plot')
    plt.show()

ticker = 'AAPL' #change ticker to see data from different stocks
processed_data = process_stock_data(ticker, '2010-01-01', '2024-01-01')
plot_box(processed_data['Close'])

#removing outliers with IQR
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_cleaned = data[(data >= lower_bound) & (data <= upper_bound)]
    return data_cleaned

cleaned_data = processed_data.copy()
cleaned_data['Close'] = remove_outliers(cleaned_data['Close'])
cleaned_data.dropna(inplace=True)  # removes rows with NaN values resulting from outlier removal
print(cleaned_data.head())

#perfroms k-means clustering
def perform_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data[['Close', 'SMA_50', 'SMA_200', 'MACD', 'Signal_Line']].dropna())
    return data

clustered_data = perform_clustering(cleaned_data)
print(clustered_data.head())

# Visualize Clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(data=clustered_data, x='Close', y='SMA_50', hue='Cluster', palette='viridis')
plt.title('Clusters Visualization')
plt.show()

"""To identify outliers in the dataset, we first need to use the IQR method. To do this we calculate the first quartile (Q1) and third quartile (Q3). Datapoints outside these bounds are considered outliers. After finding our outliers, we remove them from the dataset which would increase our model's accuracy as well as better clustering and insights on the clusters."""

#process data
ticker = 'AAPL' #change ticker to see data from different stocks
seq_length = 60
processed_data = process_stock_data(ticker, '2010-01-01', '2024-01-01')

#clean data
data = processed_data.copy()
data['Close'] = remove_outliers(data['Close'])
data.dropna(inplace=True)

#split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

#prepare variables
exog_features = ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line']
train_exog = train_data[exog_features]
test_exog = test_data[exog_features]

#hyperparameter tuning using grid search
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

print(f"Best ARIMAX order: {best_order} with AIC: {best_aic}")

# Use the best model to make predictions
predictions = best_mdl.forecast(steps=len(test_data), exog=test_exog)
predictions.index = test_data.index

#calculate error
mse = mean_squared_error(test_data['Close'], predictions)
print(f'Mean Squared Error: {mse}')

#plot results
plt.figure(figsize=(14, 5))
plt.plot(train_data.index, train_data['Close'], label='Train')
plt.plot(test_data.index, test_data['Close'], label='Test')
plt.plot(predictions.index, predictions, label='Predicted')
plt.title('Stock Price Prediction of ' + ticker + ' using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

"""Here I used a grid search to find the best p, d, and q values for the ARIMA parameters. I did this by fitting every possible p, d, and q values (from ranges 0-6) and finding which model had the highest accuracy with those parameters.

User Interaction and Feedback:
- Improve the user interaction part of the code to accept multiple stock tickers and provide multiple recommendations by modifying the input gathering step.
- Implement a method for collecting user feedback by incorporating an interactive prompt after recommendations are made.
"""

def process_stock_data(tickers, start_date, end_date):
  # Using tickers user can input multiple tickers of their choosing

  stockData = {}
  #For loop it iterate over each ticker that the user has entered in tickers

  for ticker in tickers:
    data = yf.download(ticker, start = start_date, end = end_date)
    closing_prices = data['Close']
    stockData[ticker] = closing_prices
    # Returns the closing prices

    return stockData

# Using another function(get_user_feedback) to get feedback from the users
def get_user_feedback(recommendations):
  total_feedback = {}
  for ticker in recommendations:
    # Has the user rate the recommendation from 1-10

    user_input = int(input(f"Rate the recommendation for {ticker} on a scale from 1-10: "))
    # Makes sure they entered a correct value with the if-else statement

    if user_input >= 1 or user_input <= 10:
      total_feedback[ticker] = user_input

    else:
      user_input = int(input(f"Please input a number between 1-10 for your rating"))

    # Returns all of the feedback which can be used to see what the user thinks about our recommendations and will allow us to make changes as necessary

  return total_feedback

"""To summarize: In order to be able to process multiple recommendations in the function process_stock_data, I used an empty dictionary(stockData), and iterated through the tickers the user had inputted to get their closing prices. Then to get recommendations I used a separate function which got the user's recommendation and processed it into an empty dictionary. This dictionary is then returned and we can use the ratings the user has left in order to modify our program's recommendation process and accuracy."""

