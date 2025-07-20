# GOOGL-Stock-Data-Visualization

## 📈 Visualizing Alphabet Inc. (GOOGL) Stock Performance

This repository contains a historical stock price dataset for Alphabet Inc.'s Class A shares (GOOGL) and a Python script to generate comprehensive visualizations of its market activity. The visualizations include raw price movements, common moving averages, Bollinger Bands for volatility, and the Relative Strength Index (RSI) for momentum.

This project serves as an excellent starting point for understanding financial time series data, performing exploratory data analysis (EDA), and visualizing key technical indicators.

### 📊 Dataset: `GoogleStockPrices.csv`

**Context:**
In the dynamic world of finance, historical stock data serves as a cornerstone for analysis, predictive modeling, and strategic investment decisions. Understanding the past performance of market leaders like Alphabet Inc.'s Class A shares (GOOGL) is crucial for identifying trends, assessing volatility, and developing robust forecasting models. This dataset offers a detailed look into nearly a decade of GOOGL's daily market activity.

**Overview:**
This dataset provides daily historical stock price and trading volume data for Alphabet Inc.'s Class A (voting) shares, traded under the ticker symbol GOOGL.

| Category      | Detail                                        |
| :------------ | :-------------------------------------------- |
| **Records** | 2500+ daily observations (trading days only)  |
| **Timeframe** | January 1, 2015 - December 31, 2024           |
| **Company** | Alphabet Inc. (GOOGL)                         |
| **Data Points** | `Open`, `High`, `Low`, `Close`, `Volume`      |
| **Frequency** | Daily (trading days only)                     |
| **Source** | Derived directly from Google Finance          |

**Columns:**

* **`Date`**: The trading date (YYYY-MM-DD format).
* **`Open`**: The opening price of the stock on that trading day.
* **`High`**: The highest price reached during that trading day.
* **`Low`**: The lowest price reached during that trading day.
* **`Close`**: The closing price of the stock on that trading day.
* **`Volume`**: The total number of shares traded on that day.

**Key Features:**

* **Extensive Temporal Coverage**: Nearly a decade of market data for GOOGL, from 2015 through 2024.
* **Single Ticker Focus**: Dedicated to Alphabet Inc.'s Class A (GOOGL) shares, allowing for in-depth analysis of this specific stock.
* **Comprehensive Daily Metrics**: Includes Open, High, Low, Close prices, and trading Volume, essential for detailed technical analysis.
* **Trading Day Focus**: Data is strictly limited to actual market trading days, reflecting real market activity without artificial interpolations for non-trading days.
* **Direct Source**: Data is derived directly from Google Finance, a reliable and widely recognized source for financial information.

**Uses:**

This dataset is ideally suited for a wide range of financial and data science applications, including:

* Time Series Analysis: Analyze long-term price trends, seasonality, and cycles in GOOGL's stock performance.
* Predictive Modeling: Develop and test models for daily GOOGL stock price forecasting, volatility prediction, or trend identification.
* Algorithmic Trading Strategies: Backtest daily trading algorithms based on GOOGL's historical price and volume data.
* Risk Assessment: Study historical volatility and drawdowns of GOOGL to understand potential investment risks.
* Market Event Analysis: Examine the daily impact of major corporate announcements, economic news, or global events specifically on GOOGL's stock.

### 💻 Data Visualization Code: `stock_data_visualizer.py`

This Python script loads the `GoogleStockPrices.csv` dataset, calculates various technical indicators, and generates an interactive multi-panel plot to visualize the stock's performance.

#### **Python Libraries Used:**

* `pandas`: For data manipulation and analysis.
* `matplotlib`: For creating static, interactive, and animated visualizations.
* `seaborn`: For enhancing the aesthetics and statistical plots.
* `numpy`: For numerical operations.

#### **Explanation of Each Graph:**

The script generates a multi-panel plot, typically consisting of 2 or 3 subplots depending on the configuration:

1.  **Main Price Plot (Top Panel)**
    * **Content**: This is the primary display for the stock's price movements.
        * **High-Low Range (Gray Shaded Area)**: Represents the daily price range, from the lowest (`Low`) to the highest (`High`) price reached. This visually mimics the "wicks" of candlestick charts, showing the volatility within each day.
        * **Open Price (Orange Dashed Line)**: The price at which the stock started trading on a given day.
        * **Close Price (Blue Solid Line)**: The final price at which the stock traded on a given day. This is often considered the most important price for daily analysis.
        * **Moving Averages (Purple Dashed/Green Dotted Lines)**:
            * **20-Day Moving Average (MA20)**: The average closing price over the past 20 trading days. It provides a short-term trend indication.
            * **50-Day Moving Average (MA50)**: The average closing price over the past 50 trading days. It indicates a medium-term trend.
            * **200-Day Moving Average (MA200)**: (If enabled in `CONFIG`) The average closing price over the past 200 trading days. It represents a significant long-term trend.
            * *Inference*: Moving averages smooth out price fluctuations, making it easier to identify trends. Crossovers between shorter and longer MAs can signal potential trend changes.
        * **Bollinger Bands (Red Dotted Lines & Shaded Area)**: (If enabled in `CONFIG`)
            * Consist of a middle band (typically a 20-day Simple Moving Average), an upper band (2 standard deviations above the middle band), and a lower band (2 standard deviations below).
            * *Inference*: Bollinger Bands measure market volatility. When bands widen, volatility increases; when they narrow, volatility decreases. Prices tend to stay within the bands, and touches/breaks of the bands can signal overbought/oversold conditions or potential reversals.

2.  **Volume Plot (Middle Panel - Optional)**
    * **Content**: A bar chart showing the daily trading `Volume` (number of shares traded).
    * *Inference*: Volume is crucial for confirming price trends. High volume on a price move (up or down) indicates strong conviction behind that move, while low volume might suggest a weak trend.

3.  **RSI Plot (Bottom Panel - Optional)**
    * **Content**: A line plot of the **Relative Strength Index (RSI)**, typically a 14-period momentum oscillator.
    * **Horizontal Lines**: Includes a dashed red line at 70 (indicating **overbought** conditions) and a dashed green line at 30 (indicating **oversold** conditions).
    * *Inference*: RSI measures the speed and change of price movements. It oscillates between 0 and 100. Readings above 70 suggest the asset is overbought (potentially due for a pullback), while readings below 30 suggest it is oversold (potentially due for a rebound).

#### **Configuration Options (`CONFIG` in `stock_data_visualizer.py`):**

The script allows for easy customization through a `CONFIG` dictionary:

* `file_path`: Path to your CSV data.
* `company_name`: Name of the stock (e.g., 'GOOGL') for plot titles.
* `title_font_size`, `label_font_size`, `legend_font_size`, `figure_size`: Control plot aesthetics.
* `ma_windows`: List of moving average periods to calculate and plot (e.g., `[20, 50, 200]`).
* `enable_volume_plot`: `True`/`False` to show/hide the volume subplot.
* `enable_bollinger_bands`: `True`/`False` to show/hide Bollinger Bands.
* `enable_rsi`: `True`/`False` to show/hide the RSI subplot.
* `start_date`, `end_date`: Filter the data to a specific date range for plotting.

