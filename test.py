import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Fetch historical data for NVDA
symbol = 'TSLA'
start_date = '2023-01-01'
data = yf.download(symbol, start=start_date)

# Calculate moving averages
data['10_MA'] = data['Close'].rolling(window=10).mean()
data['20_MA'] = data['Close'].rolling(window=20).mean()  # Fixed variable name from 10_MA
data['50_MA'] = data['Close'].rolling(window=50).mean()

# Generate signals - use loc to avoid chained indexing
data['Signal'] = 0
data.loc[data.index[20:], 'Signal'] = (data['20_MA'][20:] > data['50_MA'][20:]).astype(int)
data['Position'] = data['Signal'].diff()

# Print number of signals to verify they exist
buy_signals = data[data['Position'] == 1]
sell_signals = data[data['Position'] == -1]
print(f"Number of buy signals: {len(buy_signals)}")
print(f"Number of sell signals: {len(sell_signals)}")

# Plot with interactive checkboxes
fig, ax = plt.subplots(figsize=(14, 6))
lines = []
labels = []

# Plot close price (always visible)
l_close, = ax.plot(data['Close'], label='Close Price', alpha=0.5)
lines.append(l_close)
labels.append('Close Price')

# Plot moving averages
l_10ma, = ax.plot(data['10_MA'], label='10-Day MA', visible=False)
l_20ma, = ax.plot(data['20_MA'], label='20-Day MA', visible=True)
l_50ma, = ax.plot(data['50_MA'], label='50-Day MA', visible=True)

lines.extend([l_10ma, l_20ma, l_50ma])
labels.extend(['10-Day MA', '20-Day MA', '50-Day MA'])

# Only plot signals if they exist
if not buy_signals.empty:
    l_buy = ax.plot(buy_signals.index, data.loc[buy_signals.index, '20_MA'], '^', 
             color='green', label='Buy Signal', markersize=10)[0]
    lines.append(l_buy)
    labels.append('Buy Signal')
    
if not sell_signals.empty:
    l_sell = ax.plot(sell_signals.index, data.loc[sell_signals.index, '20_MA'], 'v', 
             color='red', label='Sell Signal', markersize=10)[0]
    lines.append(l_sell)
    labels.append('Sell Signal')

ax.set_title(f'{symbol}: Moving Average Crossover')
ax.grid(True)

# Calculate investment returns
def calculate_returns(data, buy_signals, sell_signals, initial_investment=1000):
    # Trading strategy portfolio
    portfolio = pd.DataFrame(index=data.index)
    portfolio['holdings'] = 0  # Shares held
    portfolio['cash'] = 0  # Cash on hand
    portfolio['price'] = data['Close']  # Share price
    portfolio['position'] = data['Position']  # Buy/sell signals
    
    # Process all trading days
    # Start by investing everything on the first day
    initial_price = portfolio['price'].iloc[0]
    current_shares = initial_investment / initial_price
    cash_balance = 0
    
    for date in portfolio.index:
        # If this is a buy signal day and we have cash, buy shares
        if date in buy_signals.index and cash_balance > 0:
            price = portfolio.loc[date, 'price']
            # Invest all available cash
            shares_to_buy = cash_balance / price
            current_shares += shares_to_buy
            cash_balance = 0
        
        # If this is a sell signal day and we have shares, sell all
        elif date in sell_signals.index and current_shares > 0:
            price = portfolio.loc[date, 'price']
            # Sell all shares
            cash_balance += current_shares * price
            current_shares = 0
        
        # Update portfolio values for this day
        portfolio.loc[date, 'holdings'] = current_shares
        portfolio.loc[date, 'cash'] = cash_balance
    
    # Calculate total value for each day (cash + holdings value)
    portfolio['holdings_value'] = portfolio['holdings'] * portfolio['price']
    portfolio['total_value'] = portfolio['cash'] + portfolio['holdings_value']
    
    # Get final stats
    final_value = portfolio['total_value'].iloc[-1]
    total_return = final_value - initial_investment
    percent_return = (total_return / initial_investment) * 100
    
    # Buy and hold strategy
    hold_portfolio = pd.DataFrame(index=data.index)
    hold_portfolio['price'] = data['Close']
    
    # Buy at first available date
    initial_price = hold_portfolio['price'].iloc[0]
    shares_bought = initial_investment / initial_price
    
    # Calculate value over time
    hold_portfolio['value'] = hold_portfolio['price'] * shares_bought
    
    # Get final stats for buy and hold
    hold_final_value = hold_portfolio['value'].iloc[-1]
    hold_total_return = hold_final_value - initial_investment
    hold_percent_return = (hold_total_return / initial_investment) * 100
    
    return (portfolio, final_value, total_return, percent_return, 
            hold_portfolio, hold_final_value, hold_total_return, hold_percent_return)

if len(buy_signals) > 0 or len(sell_signals) > 0:
    (portfolio, final_value, total_return, percent_return,
     hold_portfolio, hold_final_value, hold_total_return, hold_percent_return) = calculate_returns(data, buy_signals, sell_signals)
    
    # Display results
    print(f"\nInvestment Results:")
    print(f"Initial Investment: ${1000:.2f}")
    print(f"\nTrading Strategy:")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: ${total_return:.2f}")
    print(f"Percent Return: {percent_return:.2f}%")
    
    print(f"\nBuy and Hold Strategy:")
    print(f"Final Value: ${hold_final_value:.2f}")
    print(f"Total Return: ${hold_total_return:.2f}")
    print(f"Percent Return: {hold_percent_return:.2f}%")
    
    print(f"\nComparison:")
    difference = total_return - hold_total_return
    if difference > 0:
        print(f"Trading strategy outperformed buy & hold by ${difference:.2f} ({(difference/hold_total_return*100):.2f}%)")
    else:
        print(f"Buy & hold outperformed trading strategy by ${-difference:.2f} ({(-difference/total_return*100):.2f}%)")
    
    # Add portfolio value plots
    ax2 = ax.twinx()
    l_portfolio, = ax2.plot(portfolio['total_value'], 'purple', alpha=0.7, linewidth=1.5, label='Trading Strategy')
    l_hold, = ax2.plot(hold_portfolio['value'], 'blue', alpha=0.7, linewidth=1.5, label='Buy & Hold')
    ax2.set_ylabel('Portfolio Value ($)', color='purple')
    
    # Add to lines and labels
    lines.extend([l_portfolio, l_hold])
    labels.extend(['Trading Strategy', 'Buy & Hold'])
else:
    print("\nNo trading signals detected. Cannot calculate returns.")

# Add checkboxes for toggling visibility
rax = plt.axes([0.02, 0.5, 0.12, 0.15])
visibility = [line.get_visible() for line in lines]
check = CheckButtons(rax, labels, visibility)

def toggle_visibility(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(toggle_visibility)

plt.tight_layout()
plt.show()
