#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import yfinance as yf

# List of top 20 stocks by market capitalization (you'll need to update this list)
top_20_stocks = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS',
    'ITC.NS', 'BAJFINANCE.NS', 'LTI.NS', 'MARUTI.NS', 'NTPC.NS',
    'TATAMOTORS.NS', 'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 
    'AXISBANK.NS'
]

# Create a DataFrame to store stock symbols and their details
df_stocks = pd.DataFrame(top_20_stocks, columns=['Stock Symbol'])

# Fetch historical data for each stock and add details
def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, start='2019-01-01', end='2024-01-01')
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Fetch data and populate DataFrame
stock_details = []
for symbol in top_20_stocks:
    data = fetch_stock_data(symbol)
    if not data.empty:
        market_cap = None  # Market cap needs to be fetched from another source
        # Append details for each stock
        stock_details.append({
            'Stock Symbol': symbol,
            'Market Capitalization (in Crores)': market_cap,  # Placeholder for actual market cap
            'Recent Close Price': data['Close'].iloc[-1] if not data['Close'].empty else None
        })

df_stocks = pd.DataFrame(stock_details)

# Print the table
print(df_stocks.to_string(index=False))


# In[25]:


data


# In[55]:


import yfinance as yf
from tabulate import tabulate

# Download TCS data from Yahoo Finance
tcs_data = yf.download('TCS.NS', start='2019-01-01', end='2024-01-01')

# Reset the index to make Date a column instead of the index
tcs_data.reset_index(inplace=True)

# Display the first few rows of the data in a table format
print(tabulate(tcs_data.head(), headers='keys', tablefmt='psql'))


# In[57]:


import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import os
from datetime import date
from nsepy import get_history
import yfinance as yf

warnings.simplefilter(action='ignore', category=Warning)

# Task 1: Data Acquisition and Preparation
# Define the stock symbols
scripts = ['RELIANCE', 'TCS', 'HDFCBANK', 'BHARTIARTL', 'ICICIBANK', 'INFY', 'SBIN', 'LICI', 'ITC', 'LT',
           'HCLTECH', 'BAJFINANCE', 'ONGC', 'SUNPHARMA', 'MARUTI', 'NTPC', 'TATAMOTORS', 'AXISBANK', 'KOTAKBANK', 'ADANIENT', 'ULTRACEMCO']

# Download data for each stock and save as CSV
for stock in scripts:
    data = yf.download(f'{stock}.NS', start='2018-01-01', end=date.today().strftime('%Y-%m-%d'))
    data.to_csv(f'./Data/{stock}.csv')

# Task 2: Strategy Implementation and Backtesting
# Define the Golden Cross strategy function
def GoldenCrossoverSignal(name):
    path = f'./Data/{name}.csv'
    if not os.path.exists(path):
        print(f"File not found for {name}")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

    data = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    data['Prev_Close'] = data.Close.shift(1)
    data['20_SMA'] = data.Prev_Close.rolling(window=20, min_periods=1).mean()
    data['50_SMA'] = data.Prev_Close.rolling(window=50, min_periods=1).mean()
    data['Signal'] = 0
    data['Signal'] = np.where(data['20_SMA'] > data['50_SMA'], 1, 0)
    data['Position'] = data.Signal.diff()

    df_pos = data[(data['Position'] == 1) | (data['Position'] == -1)].copy()
    df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    return df_pos

# Define the Backtest class
class Backtest:
    def __init__(self):
        self.columns = ['Equity Name', 'Trade', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Quantity', 'Position Size', 'PNL', '% PNL']
        self.backtesting = pd.DataFrame(columns=self.columns)

    def buy(self, equity_name, entry_time, entry_price, qty):
        self.trade_log = dict(zip(self.columns, [None] * len(self.columns)))
        self.trade_log['Trade'] = 'Long Open'
        self.trade_log['Quantity'] = qty
        self.trade_log['Position Size'] = round(self.trade_log['Quantity'] * entry_price, 3)
        self.trade_log['Equity Name'] = equity_name
        self.trade_log['Entry Time'] = entry_time
        self.trade_log['Entry Price'] = round(entry_price, 2)

    def sell(self, exit_time, exit_price, exit_type, charge):
        self.trade_log['Trade'] = 'Long Closed'
        self.trade_log['Exit Time'] = exit_time
        self.trade_log['Exit Price'] = round(exit_price, 2)
        self.trade_log['Exit Type'] = exit_type
        self.trade_log['PNL'] = round((self.trade_log['Exit Price'] - self.trade_log['Entry Price']) * self.trade_log['Quantity'] - charge, 3)
        self.trade_log['% PNL'] = round((self.trade_log['PNL'] / self.trade_log['Position Size']) * 100, 3)
        self.trade_log['Holding Period'] = exit_time - self.trade_log['Entry Time']
        self.backtesting = pd.concat([self.backtesting, pd.DataFrame([self.trade_log])], ignore_index=True)

    def stats(self):
        df = self.backtesting
        parameters = ['Total Trade Scripts', 'Total Trade', 'PNL',  'Winners', 'Losers', 'Win Ratio','Total Profit', 'Total Loss', 'Average Loss per Trade', 'Average Profit per Trade', 'Average PNL Per Trade', 'Risk Reward']
        total_traded_scripts = len(df['Equity Name'].unique())
        total_trade = len(df.index)
        pnl = df.PNL.sum()
        winners = len(df[df.PNL > 0])
        loosers = len(df[df.PNL <= 0])
        win_ratio = str(round((winners/total_trade) * 100, 2)) + '%'
        total_profit = round(df[df.PNL > 0].PNL.sum(), 2)
        total_loss  = round(df[df.PNL <= 0].PNL.sum(), 2)
        average_loss_per_trade = round(total_loss/loosers, 2)
        average_profit_per_trade = round(total_profit/winners, 2)
        average_pnl_per_trade = round(pnl/total_trade, 2)
        risk_reward = f'1:{-1 * round(average_profit_per_trade/average_loss_per_trade, 2)}'
        data_points = [total_traded_scripts,total_trade,pnl,winners, loosers, win_ratio, total_profit, total_loss, average_loss_per_trade, average_profit_per_trade, average_pnl_per_trade, risk_reward]
        data = list(zip(parameters,data_points ))
        print(tabulate(data, ['Parameters', 'Values'], tablefmt='psql')) 

# Run the backtest
bt = Backtest()
capital = 50000

for stock in scripts:
    data = GoldenCrossoverSignal(stock)
    if data.empty:
        continue  # Skip if no data is returned

    buy_signals = data[data['Position'] == 'Buy']
    sell_signals = data[data['Position'] == 'Sell']

    if buy_signals.empty or sell_signals.empty:
        continue  # Skip if there are no buy or sell signals

    required_df = data[(data.index >= buy_signals.index[0]) & (data.index <= sell_signals.index[-1])]
    for index, row in required_df.iterrows():
        if row['Position'] == 'Buy':
            qty = capital // row['Open']
            bt.buy(stock, index, row['Open'], qty)
        elif row['Position'] == 'Sell':
            bt.sell(index, row['Open'], 'Exit Trigger', 0)

bt.backtesting.PNL.sum()
bt.backtesting.to_csv('Backtest.csv')
bt.stats()

# Task 3: Analysis and Insights
# Visualization function for a specific stock
def visualize_signals(stock):
    data = pd.read_csv(f'./Data/{stock}.csv', parse_dates=['Date'], index_col='Date')
    data['20_SMA'] = data.Close.rolling(window=20, min_periods=1).mean()
    data['50_SMA'] = data.Close.rolling(window=50, min_periods=1).mean()
    data['Signal'] = 0
    data['Signal'] = np.where(data['20_SMA'] > data['50_SMA'], 1, 0)
    data['Position'] = data.Signal.diff()

    plt.figure(figsize=(20,10))
    data['Close'].plot(color='k', label='Close Price')
    data['20_SMA'].plot(color='b', label='20-day SMA')
    data['50_SMA'].plot(color='g', label='50-day SMA')
    plt.plot(data[data['Position'] == 1].index, data['20_SMA'][data['Position'] == 1], '^', markersize=15, color='g', label='buy')
    plt.plot(data[data['Position'] == -1].index, data['20_SMA'][data['Position'] == -1], 'v', markersize=15, color='r', label='sell')
    plt.ylabel('Price in Rupees', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.title(stock, fontsize=20)
    plt.savefig(stock)
    plt.legend()
    plt.grid()
    plt.show()

# Visualize signals for each stock
for stock in scripts:
    visualize_signals(stock)


# In[58]:


import matplotlib.pyplot as plt

# Plotting Total PNL by Stock
def plot_total_pnl_by_stock(backtest_df):
    pnl_by_stock = backtest_df.groupby('Equity Name')['PNL'].sum()
    
    plt.figure(figsize=(12, 8))
    pnl_by_stock.plot(kind='bar', color='skyblue')
    plt.title('Total PNL by Stock')
    plt.savefig('Total PNL by Stock.jpg')
    plt.xlabel('Stock')
    plt.ylabel('Total PNL')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Assuming `bt.backtesting` is the DataFrame from the Backtest class
plot_total_pnl_by_stock(bt.backtesting)


# In[59]:


import pandas as pd
import matplotlib.pyplot as plt

def plot_top_20_stocks_by_pnl(backtest_df):
    # Group by stock and sum the PNL
    pnl_by_stock = backtest_df.groupby('Equity Name')['PNL'].sum()
    
    # Sort by PNL and select top 20
    top_20_pnl = pnl_by_stock.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    top_20_pnl.plot(kind='bar', color='skyblue')
    plt.title('Top 20 Stocks by Total PNL')
    plt.xlabel('Stock')
    plt.ylabel('Total PNL')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Call the function
plot_top_20_stocks_by_pnl(bt.backtesting)


# In[60]:


def plot_top_20_stocks_by_trades(backtest_df):
    # Count trades per stock
    trades_by_stock = backtest_df['Equity Name'].value_counts()
    
    # Select top 20 stocks with the most trades
    top_20_trades = trades_by_stock.head(20)
    
    plt.figure(figsize=(12, 8))
    top_20_trades.plot(kind='bar', color='lightgreen')
    plt.title('Top 20 Stocks by Number of Trades')
    plt.xlabel('Stock')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Call the function
plot_top_20_stocks_by_trades(bt.backtesting)


# In[30]:


def plot_top_20_stocks_by_avg_pnl(backtest_df):
    # Calculate average PNL per trade for each stock
    avg_pnl_by_stock = backtest_df.groupby('Equity Name')['PNL'].mean()
    
    # Sort by average PNL and select top 20
    top_20_avg_pnl = avg_pnl_by_stock.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    top_20_avg_pnl.plot(kind='bar', color='lightcoral')
    plt.title('Top 20 Stocks by Average PNL Per Trade')
    plt.savefig('Top 20 Stocks by Average PNL Per Trade')
    plt.xlabel('Stock')
    plt.ylabel('Average PNL')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Call the function
plot_top_20_stocks_by_avg_pnl(bt.backtesting)


# In[61]:


def calculate_performance_metrics(backtest_df):
    # Total Returns
    total_return = backtest_df['PNL'].sum()
    
    # Annualized Returns
    total_days = (backtest_df['Exit Time'].max() - backtest_df['Entry Time'].min()).days
    annualized_return = (1 + (total_return / backtest_df['Position Size'].sum()))**(365 / total_days) - 1

    # Maximum Drawdown
    cumulative_returns = (backtest_df['PNL'].cumsum() / backtest_df['Position Size'].sum()) + 1
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio
    risk_free_rate = 0.05  # Example risk-free rate
    avg_daily_pnl = backtest_df['PNL'].mean()
    daily_std_pnl = backtest_df['PNL'].std()
    sharpe_ratio = (avg_daily_pnl - risk_free_rate / 252) / daily_std_pnl * np.sqrt(252)

    # Win/Loss Ratio
    total_trades = len(backtest_df)
    wins = len(backtest_df[backtest_df['PNL'] > 0])
    losses = len(backtest_df[backtest_df['PNL'] <= 0])
    win_loss_ratio = wins / losses if losses != 0 else np.nan

    # Prepare results for plotting
    metrics = {
        'Total Returns': total_return,
        'Annualized Returns': annualized_return,
        'Maximum Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Win/Loss Ratio': win_loss_ratio,
        'Number of Trades': total_trades
    }
    
    return metrics


# In[62]:


def calculate_yearly_metrics(backtest_df):
    # Ensure 'Entry Time' and 'Exit Time' are datetime
    backtest_df['Entry Time'] = pd.to_datetime(backtest_df['Entry Time'])
    backtest_df['Exit Time'] = pd.to_datetime(backtest_df['Exit Time'])
    
    # Extract year
    backtest_df['Year'] = backtest_df['Entry Time'].dt.year
    
    # Group by year
    yearly_summary = backtest_df.groupby('Year').apply(calculate_performance_metrics).apply(pd.Series)
    
    return yearly_summary

def plot_yearly_metrics(yearly_metrics):
    # Plot metrics for each year
    plt.figure(figsize=(15, 10))
    
    for metric in yearly_metrics.columns:
        plt.plot(yearly_metrics.index, yearly_metrics[metric], marker='o', label=metric)
    
    plt.title('Yearly Performance Metrics')
    plt.savefig('Yearly Performance Metrics.jpg')
    plt.xlabel('Year')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Calculate yearly metrics
yearly_metrics = calculate_yearly_metrics(bt.backtesting)

# Plot yearly metrics
plot_yearly_metrics(yearly_metrics)


# In[38]:


# Define the tickers and the time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 
    'TATAMOTORS.NS', 'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 
    'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Initialize dictionary to store total returns
total_returns = {}

# Download data and calculate total returns for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    total_return = (end_price - start_price) / start_price * 100
    total_returns[ticker] = total_return

# Prepare data for plotting
names = list(total_returns.keys())
values = list(total_returns.values())

# Plot the bar chart
plt.figure(figsize=(16, 9))
plt.bar(names, values, color='blue')
plt.xlabel('Stock')
plt.ylabel('Total Return (%)')
plt.title('Total Return of Selected Stocks (2018-2023)')
plt.savefig('Total Return of Selected Stocks (2018-2023).jpg')
plt.ylim(0, max(values) + 10)  # Adjust y-axis limit for better visibility of bars
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[9]:


# Define the tickers and the time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 
    'TATAMOTORS.NS', 'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 
    'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Initialize dictionary to store annualized returns
annualized_returns = {}

# Download data and calculate annualized returns for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    total_return = (end_price - start_price) / start_price
    num_years = (stock_data.index[-1] - stock_data.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    annualized_returns[ticker] = annualized_return * 100

# Prepare data for plotting
names = list(annualized_returns.keys())
values = list(annualized_returns.values())

# Plot the bar chart
plt.figure(figsize=(14, 8))
plt.bar(names, values, color='blue')
plt.xlabel('Stock')
plt.ylabel('Annualized Return (%)')
plt.title('Annualized Return of Selected Stocks (2018-2023)')
plt.savefig('Annualized Return of Selected Stocks (2018-2023).jpg')
plt.ylim(0, max(values) + 10)  # Adjust y-axis limit for better visibility of bars
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[10]:


# Define the tickers and the time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS', 
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Function to calculate maximum drawdown
def calculate_max_drawdown(prices):
    drawdowns = (prices / prices.cummax()) - 1
    max_drawdown = drawdowns.min()
    return max_drawdown

# Initialize dictionary to store maximum drawdowns
max_drawdowns = {}

# Download data and calculate maximum drawdown for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    max_drawdown = calculate_max_drawdown(stock_data['Close'])
    max_drawdowns[ticker] = max_drawdown * 100

# Prepare data for plotting
names = list(max_drawdowns.keys())
values = list(max_drawdowns.values())

# Plot the bar chart
plt.figure(figsize=(14, 8))
plt.bar(names, values, color='red')
plt.xlabel('Stock')
plt.ylabel('Maximum Drawdown (%)')
plt.title('Maximum Drawdown of Selected Stocks (2018-2023)')
plt.savefig('Maximum Drawdown of Selected Stocks (2018-2023).jpg')
plt.ylim(min(values) - 10, 0)  # Adjust y-axis limit for better visibility of bars
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[51]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS', 
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'
risk_free_rate = 0.06  # Example risk-free rate of 6% per year

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate / 252  # Annualized risk-free rate converted to daily
    annualized_excess_return = excess_returns.mean() * 252  # Annualize excess return
    annualized_volatility = excess_returns.std() * np.sqrt(252)  # Annualize volatility
    if annualized_volatility == 0:
        return 0  # Avoid division by zero
    return annualized_excess_return / annualized_volatility

# Initialize dictionary to store Sharpe Ratios
sharpe_ratios = {}

# Download data and calculate Sharpe Ratio for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    sharpe_ratio = calculate_sharpe_ratio(stock_data['Daily Return'].dropna(), risk_free_rate)
    sharpe_ratios[ticker] = sharpe_ratio

# Prepare data for plotting
names = list(sharpe_ratios.keys())
values = list(sharpe_ratios.values())

# Plot the bar chart
plt.figure(figsize=(14, 8))
plt.bar(names, values, color='green')
plt.xlabel('Stock')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio of Selected Stocks (2018-2023)')
plt.savefig('Sharpe Ratio of Selected Stocks (2018-2023).jpg')
plt.ylim(0, max(values) + 0.5)  # Adjust y-axis limit for better visibility of bars
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[21]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS', 
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Function to calculate Win/Loss Ratio
def calculate_win_loss_ratio(returns):
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    if losses == 0:
        return np.inf  # Avoid division by zero
    return wins / losses

# Initialize dictionary to store Win/Loss Ratios
win_loss_ratios = {}

# Download data and calculate Win/Loss Ratio for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    win_loss_ratio = calculate_win_loss_ratio(stock_data['Daily Return'].dropna())
    win_loss_ratios[ticker] = win_loss_ratio

# Prepare data for plotting
names = list(win_loss_ratios.keys())
values = list(win_loss_ratios.values())

# Plot the line chart
plt.figure(figsize=(14, 8))
plt.plot(names, values, marker='o', linestyle='-', color='purple', markersize=8)
plt.xlabel('Stock')
plt.ylabel('Win/Loss Ratio')
plt.title('Win/Loss Ratio of Selected Stocks (2018-2023)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()


# In[22]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS',
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS',
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Initialize dictionary to store the number of trades
number_of_trades = {}

# Download data and calculate the number of trades for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    num_trades = stock_data['Daily Return'].dropna().ne(0).sum()  # Count non-zero returns
    number_of_trades[ticker] = num_trades

# Convert data to DataFrame for easier handling
df = pd.DataFrame(list(number_of_trades.items()), columns=['Stock', 'Number of Trades'])

# Create Waterfall Chart
def plot_waterfall(data):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate the cumulative number of trades
    data['Cumulative Trades'] = data['Number of Trades'].cumsum()

    # Plot initial bar
    ax.bar(data['Stock'][0], data['Cumulative Trades'][0], color='blue')

    # Plot subsequent bars
    for i in range(1, len(data)):
        previous_value = data['Cumulative Trades'][i - 1]
        current_value = data['Cumulative Trades'][i]
        ax.bar(data['Stock'][i], current_value - previous_value, bottom=previous_value, color='orange')

    # Add labels
    for i, row in data.iterrows():
        ax.text(row['Stock'], row['Cumulative Trades'], f'{row["Number of Trades"]}', ha='center', va='bottom')

    ax.set_xlabel('Stock')
    ax.set_ylabel('Number of Trades Executed')
    ax.set_title('Number of Trades Executed for Selected Stocks (2018-2023)')
    plt.savefig('Number of Trades Executed for Selected Stocks (2018-2023).jpg')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Plot the waterfall chart
plot_waterfall(df)


# In[23]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS',
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS',
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Function to calculate yearly metrics
def calculate_metrics(df):
    df = df.copy()
    df['Year'] = df.index.year
    yearly_metrics = df.groupby('Year').agg(
        Total_Return=('Close', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100),
        Annualized_Returns=('Close', lambda x: (x.iloc[-1] / x.iloc[0]) ** (1 / len(x.index.year.unique())) - 1),
        Maximum_Drawdown=('Close', lambda x: (x / x.cummax() - 1).min() * 100),
        Sharpe_Ratio=('Close', lambda x: x.pct_change().mean() / x.pct_change().std() * np.sqrt(252)),
        Win_Loss_Ratio=('Close', lambda x: (x.pct_change() > 0).sum() / (x.pct_change() < 0).sum() if (x.pct_change() < 0).sum() > 0 else np.nan),
        Number_of_Trades=('Close', lambda x: (x.pct_change() != 0).sum())
    ).reset_index()
    return yearly_metrics

# Plot the metrics for each stock
def plot_yearly_metrics(tickers):
    metrics_list = ['Total_Return', 'Annualized_Returns', 'Maximum_Drawdown', 'Sharpe_Ratio', 'Win_Loss_Ratio', 'Number_of_Trades']
    
    for metric in metrics_list:
        plt.figure(figsize=(14, 8))
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            stock_metrics = calculate_metrics(stock_data)
            plt.plot(stock_metrics['Year'], stock_metrics[metric], label=ticker)
        
        plt.xlabel('Year')
        plt.ylabel(metric)
        plt.title(f'{metric} for Selected Stocks (2018-2023)')
        plt.savefig(f'{metric} for Selected Stocks (2018-2023).jpg')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Call the function to plot yearly metrics
plot_yearly_metrics(tickers)


# In[25]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS',
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS',
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Function to calculate backtesting metrics
def calculate_metrics(df):
    df = df.copy()
    df['Year'] = df.index.year
    yearly_metrics = df.groupby('Year').agg(
        Total_Return=('Close', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100),
        Annualized_Returns=('Close', lambda x: (x.iloc[-1] / x.iloc[0]) ** (1 / len(x.index.year.unique())) - 1),
        Maximum_Drawdown=('Close', lambda x: (x / x.cummax() - 1).min() * 100),
        Sharpe_Ratio=('Close', lambda x: x.pct_change().mean() / x.pct_change().std() * np.sqrt(252)),
        Win_Loss_Ratio=('Close', lambda x: (x.pct_change() > 0).sum() / (x.pct_change() < 0).sum() if (x.pct_change() < 0).sum() > 0 else np.nan),
        Number_of_Trades=('Close', lambda x: (x.pct_change() != 0).sum())
    ).reset_index()
    return yearly_metrics

# Function to plot metrics as bar charts
def plot_metrics_bar_charts(tickers):
    metrics_list = ['Total_Return', 'Annualized_Returns', 'Maximum_Drawdown', 'Sharpe_Ratio', 'Win_Loss_Ratio', 'Number_of_Trades']
    
    for metric in metrics_list:
        plt.figure(figsize=(14, 8))
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            stock_metrics = calculate_metrics(stock_data)
            # For simplicity, we'll plot the latest year available data
            latest_year_data = stock_metrics.iloc[-1]
            plt.bar(ticker, latest_year_data[metric], label=ticker)
        
        plt.xlabel('Stock')
        plt.ylabel(metric)
        plt.title(f'{metric} for Selected Stocks (Latest Year)')
        plt.savefig(f'{metric} for Selected Stocks (Latest Year).jpg')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Call the function to plot metrics bar charts
plot_metrics_bar_charts(tickers)


# In[49]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS',
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS',
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Initialize dictionaries to store metrics
total_returns = {}
annualized_returns = {}
max_drawdown = {}
sharpe_ratio = {}
win_loss_ratio = {}
number_of_trades = {}

# Define a function to calculate metrics
def calculate_metrics(stock_data):
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    returns = stock_data['Daily Return'].dropna()
    
    # Total Returns
    total_return = (stock_data['Close'][-1] / stock_data['Close'][0]) - 1
    
    # Annualized Returns
    annualized_return = ((1 + total_return) ** (1 / (len(stock_data) / 252))) - 1
    
    # Maximum Drawdown
    rolling_max = stock_data['Close'].cummax()
    drawdown = (stock_data['Close'] - rolling_max) / rolling_max
    max_drawdown_value = drawdown.min()
    
    # Sharpe Ratio
    daily_mean = returns.mean()
    daily_std = returns.std()
    sharpe_ratio_value = (daily_mean / daily_std) * np.sqrt(252)
    
    # Win/Loss Ratio
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_loss_ratio_value = wins / losses if losses != 0 else np.inf
    
    # Number of Trades (simple approach: count of non-zero returns)
    number_of_trades_value = returns.count()
    
    return total_return, annualized_return, max_drawdown_value, sharpe_ratio_value, win_loss_ratio_value, number_of_trades_value

# Download data and calculate metrics for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    metrics = calculate_metrics(stock_data)
    
    total_returns[ticker] = metrics[0]
    annualized_returns[ticker] = metrics[1]
    max_drawdown[ticker] = metrics[2]
    sharpe_ratio[ticker] = metrics[3]
    win_loss_ratio[ticker] = metrics[4]
    number_of_trades[ticker] = metrics[5]

# Convert dictionaries to DataFrames
metrics_df = pd.DataFrame({
    'Total Returns': total_returns,
    'Annualized Returns': annualized_returns,
    'Max Drawdown': max_drawdown,
    'Sharpe Ratio': sharpe_ratio,
    'Win/Loss Ratio': win_loss_ratio,
    'Number of Trades': number_of_trades
})

# Visualization of metrics
def plot_metrics_comparison(df):
    metrics = df.columns
    num_metrics = len(metrics)
    
    fig, axs = plt.subplots((num_metrics + 1) // 2, 2, figsize=(14, 6 * ((num_metrics + 1) // 2)))
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2] if num_metrics > 1 else axs
        bars = ax.bar(df.index, df[metric], color='blue' if metric != 'Max Drawdown' else 'red')
        
        ax.set_title(metric)
        ax.set_xlabel('Stocks')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=90)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Plot metrics comparison
plot_metrics_comparison(metrics_df)


# In[50]:


# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS', 
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2018-01-01'
end_date = '2023-01-01'

# Define desired improvement targets
improvement_targets = {
    'Total Returns': 0.30,
    'Annualized Returns': 0.12,
    'Maximum Drawdown': -0.20,
    'Sharpe Ratio': 1.0,
    'Win/Loss Ratio': 1.5,
    'Number of Trades': 100
}

# Function to calculate metrics
def calculate_metrics(df):
    df['Daily Return'] = df['Close'].pct_change()
    total_return = df['Daily Return'].sum()
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    max_drawdown = (df['Close'].cummax() - df['Close']).max() / df['Close'].cummax().max()
    sharpe_ratio = df['Daily Return'].mean() / df['Daily Return'].std() * np.sqrt(252)
    win_loss_ratio = ((df['Daily Return'] > 0).sum() / (df['Daily Return'] < 0).sum()) if (df['Daily Return'] < 0).sum() > 0 else np.inf
    num_trades = df.shape[0]
    return total_return, annualized_return, max_drawdown, sharpe_ratio, win_loss_ratio, num_trades

# Initialize dictionary to store metrics
metrics = {ticker: [] for ticker in tickers}

# Download data and calculate metrics for each stock
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    metrics[ticker] = calculate_metrics(stock_data)

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics, index=['Total Returns', 'Annualized Returns', 'Maximum Drawdown', 'Sharpe Ratio', 'Win/Loss Ratio', 'Number of Trades']).T

# Calculate suggestions for improvement
def calculate_suggestions(metrics_df, targets):
    suggestions = {}
    for metric in targets.keys():
        current_values = metrics_df[metric]
        target_value = targets[metric]
        suggestions[metric] = (target_value - current_values).clip(lower=0)
    return suggestions

# Generate suggestions
suggestions = calculate_suggestions(metrics_df, improvement_targets)

# Plot the suggestions in a bar chart
def plot_suggestions(suggestions):
    metrics = list(suggestions.keys())
    values = [suggestions[m].sum() for m in metrics]  # Aggregate suggestions across all stocks

    x = np.arange(len(metrics))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(x, values, width, color='orange')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Improvement Needed')
    ax.set_title('Suggested Improvements for Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars)

    plt.tight_layout()
    plt.show()

plot_suggestions(suggestions)


# In[48]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the tickers and time period
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS',
    'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS', 'TATAMOTORS.NS',
    'ONGC.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'AXISBANK.NS'
]
start_date = '2010-01-01'
end_date = '2023-01-01'

# Download historical data
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data
    return data

data = download_data(tickers, start_date, end_date)

# Determine market conditions
def classify_market_conditions(data):
    market_conditions = {}
    for ticker, df in data.items():
        df['Return'] = df['Close'].pct_change()
        df['Cumulative Return'] = (1 + df['Return']).cumprod() - 1
        df['Monthly Return'] = df['Cumulative Return'].resample('M').ffill().pct_change()
        
        bull_market = df[df['Monthly Return'] > 0]
        bear_market = df[df['Monthly Return'] < 0]
        
        market_conditions[ticker] = {
            'Bull Market': bull_market,
            'Bear Market': bear_market
        }
    return market_conditions

market_conditions = classify_market_conditions(data)

# Calculate metrics for each market condition
def calculate_metrics(df):
    metrics = {
        'Total Returns': df['Cumulative Return'].iloc[-1],
        'Annualized Returns': df['Return'].mean() * 252,  # Assuming 252 trading days in a year
        'Maximum Drawdown': (df['Cumulative Return'] - df['Cumulative Return'].cummax()).min(),
        'Sharpe Ratio': df['Return'].mean() / df['Return'].std() * np.sqrt(252),  # Assuming 252 trading days in a year
        'Win/Loss Ratio': (df['Return'] > 0).sum() / (df['Return'] < 0).sum() if (df['Return'] < 0).sum() != 0 else np.inf
    }
    return metrics

def analyze_market_conditions(market_conditions):
    analysis = {'Bull Market': {}, 'Bear Market': {}}
    
    for ticker, conditions in market_conditions.items():
        for condition, df in conditions.items():
            metrics = calculate_metrics(df)
            if ticker not in analysis[condition]:
                analysis[condition][ticker] = metrics
                
    return analysis

analysis = analyze_market_conditions(market_conditions)

# Prepare data for plotting
def prepare_plot_data(analysis, metric_name):
    bull_values = [analysis['Bull Market'][ticker][metric_name] for ticker in tickers]
    bear_values = [analysis['Bear Market'][ticker][metric_name] for ticker in tickers]
    
    return bull_values, bear_values

def plot_metric_comparison(metric_name):
    bull_values, bear_values = prepare_plot_data(analysis, metric_name)
    
    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, bull_values, width, label='Bull Market', color='green')
    bars2 = ax.bar(x + width/2, bear_values, width, label='Bear Market', color='red')

    ax.set_xlabel('Stocks')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison: Bull vs. Bear Market')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=90)
    ax.legend()
    ax.grid(True)

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    plt.show()

# Plot metrics
plot_metric_comparison('Total Returns')
plot_metric_comparison('Annualized Returns')
plot_metric_comparison('Maximum Drawdown')
plot_metric_comparison('Sharpe Ratio')
plot_metric_comparison('Win/Loss Ratio')


# In[ ]:




