import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import seaborn as sns # For better aesthetics and potential future use

# --- Configuration Section ---
# Define file paths, company names, and plotting preferences
CONFIG = {
    'file_path': 'GoogleStockPrices.csv',
    'company_name': 'GOOGL',
    'title_font_size': 20,
    'label_font_size': 14,
    'legend_font_size': 12,
    'figure_size': (18, 10), # Slightly larger for more detail
    'ma_windows': [20, 50, 200], # Add 200-day MA
    'enable_volume_plot': True,
    'enable_bollinger_bands': True,
    'enable_rsi': True,
    'start_date': '2015-01-01', # Filter data by date
    'end_date': '2024-12-31',
}

# Ensure NLTK data is downloaded if you were to add sentiment analysis later
# import nltk
# nltk.download("stopwords")
# nltk.download("punkt")

def load_and_preprocess_data(file_path, start_date=None, end_date=None):
    """
    Loads stock data, converts date, sorts, and filters by date range.
    Handles potential file not found errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError("No data available for the specified date range. Please check your dates.")

    return df

def calculate_technical_indicators(df, ma_windows):
    """
    Calculates various technical indicators: Moving Averages, Bollinger Bands, RSI.
    """
    # Moving Averages
    for window in ma_windows:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

    # Bollinger Bands (BB)
    if CONFIG['enable_bollinger_bands']:
        window = 20 # Common window for BB
        df['BB_Mid'] = df['Close'].rolling(window=window).mean()
        df['BB_Std'] = df['Close'].rolling(window=window).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    # Relative Strength Index (RSI)
    if CONFIG['enable_rsi']:
        window = 14 # Common window for RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # Handle division by zero for rs where avg_loss is 0 (RSI becomes 100)
        df['RSI'] = df['RSI'].replace([np.inf, -np.inf], 100)
        # Handle cases where avg_gain and avg_loss are both 0 (RSI becomes 50)
        df['RSI'] = df['RSI'].fillna(50)

    return df

def plot_stock_data(df, config):
    """
    Plots the stock prices with various technical indicators and optional volume.
    """
    sns.set_style("darkgrid") # Apply seaborn style for better aesthetics

    # Determine the number of subplots
    n_subplots = 1 + config['enable_volume_plot'] + config['enable_rsi']
    fig, axes = plt.subplots(n_subplots, 1, figsize=config['figure_size'], sharex=True, gridspec_kw={'height_ratios': [3] + ([1]*(n_subplots-1))}) # Price plot is taller

    # Main Price Plot
    ax1 = axes[0] if n_subplots > 1 else axes # Handle single subplot case

    # High-Low fill to mimic candlestick wicks (visual range)
    ax1.fill_between(df['Date'], df['Low'], df['High'], color='lightgray', alpha=0.3, label='High-Low Range')

    # Price lines
    ax1.plot(df['Date'], df['Open'], label='Open', color='orange', linestyle='--', linewidth=1)
    ax1.plot(df['Date'], df['Close'], label='Close', color='blue', linewidth=2)

    # Moving Averages
    for window in config['ma_windows']:
        if f'MA{window}' in df.columns:
            ax1.plot(df['Date'], df[f'MA{window}'], label=f'{window}-Day MA', linestyle='--' if window != 200 else ':', linewidth=1.5)

    # Bollinger Bands
    if config['enable_bollinger_bands'] and 'BB_Upper' in df.columns:
        ax1.plot(df['Date'], df['BB_Upper'], label='Bollinger Upper', color='red', linestyle=':', alpha=0.7)
        ax1.plot(df['Date'], df['BB_Lower'], label='Bollinger Lower', color='red', linestyle=':', alpha=0.7)
        ax1.fill_between(df['Date'], df['BB_Lower'], df['BB_Upper'], color='red', alpha=0.1, label='Bollinger Band')

    ax1.set_title(f'{config["company_name"]} Stock Price ({df["Date"].min().year}–{df["Date"].max().year}) with Technical Indicators', fontsize=config['title_font_size'])
    ax1.set_ylabel('Price (USD)', fontsize=config['label_font_size'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=config['legend_font_size'])


    # Volume Plot
    if config['enable_volume_plot']:
        ax2 = axes[1] if n_subplots > 1 else None # Ensure ax2 exists
        if ax2 is not None:
            ax2.bar(df['Date'], df['Volume'], color='steelblue', alpha=0.7, label='Volume')
            ax2.set_ylabel('Volume', fontsize=config['label_font_size'])
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend(loc='upper left', fontsize=config['legend_font_size'])

    # RSI Plot
    if config['enable_rsi']:
        rsi_ax_idx = n_subplots - 1 # RSI is always the last subplot if enabled
        ax3 = axes[rsi_ax_idx] if n_subplots > 1 else None # Ensure ax3 exists
        if ax3 is not None:
            ax3.plot(df['Date'], df['RSI'], color='purple', label='RSI (14)', linewidth=1.5)
            ax3.axhline(70, linestyle='--', color='red', alpha=0.7, label='Overbought (70)')
            ax3.axhline(30, linestyle='--', color='green', alpha=0.7, label='Oversold (30)')
            ax3.set_ylabel('RSI', fontsize=config['label_font_size'])
            ax3.set_ylim(0, 100) # RSI range
            ax3.grid(True, linestyle='--', alpha=0.6)
            ax3.legend(loc='upper left', fontsize=config['legend_font_size'])

    # Final X-axis formatting for the bottom-most plot
    final_ax = axes[-1] if n_subplots > 1 else ax1
    final_ax.set_xlabel('Date', fontsize=config['label_font_size'])
    final_ax.xaxis.set_major_locator(mdates.YearLocator())
    final_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load and Preprocess Data
        data = load_and_preprocess_data(CONFIG['file_path'], CONFIG['start_date'], CONFIG['end_date'])

        # 2. Calculate Technical Indicators
        data = calculate_technical_indicators(data, CONFIG['ma_windows'])

        # 3. Plot Data
        plot_stock_data(data, CONFIG)

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")