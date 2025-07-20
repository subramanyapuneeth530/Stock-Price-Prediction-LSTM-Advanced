# Stock-Price-Prediction-LSTM-Advanced
An advanced stock price forecasting project using Multi-Feature, Stacked, and Bidirectional LSTMs on historical GOOGL data. Explores technical indicators, time-based features, and robust evaluation metrics.

---

## 📈 Advanced Stock Price Forecasting with Deep LSTMs

This repository presents a machine learning project focused on predicting daily stock prices using advanced Long Short-Term Memory (LSTM) neural networks. The project evolves from a basic univariate LSTM to a sophisticated multi-feature, stacked, and bidirectional LSTM architecture, incorporating various technical indicators and time-based features for enhanced predictive power.

The primary goal is to accurately forecast the next day's closing price for Alphabet Inc.'s Class A shares (GOOGL) based on historical market data.

### ✨ Key Features & Evolution

This project demonstrates an iterative approach to building robust time series forecasting models:

1.  **Basic LSTM Forecasting**:
    * **Input**: Only past `Close` prices.
    * **Goal**: Establish a baseline for predicting the next day's closing price.
    * **Learning**: Fundamental LSTM architecture and data preparation for univariate time series.

2.  **Multi-Feature LSTM Model**:
    * **Input**: `Open`, `High`, `Low`, `Close`, `Volume`.
    * **Goal**: Capture more market context and improve performance over the basic model.
    * **Learning**: Handling multivariate time series input, feature scaling.

3.  **Advanced Multi-Feature LSTM Model (Current State)**:
    * **Input**: `Open`, `High`, `Low`, `Close`, `Volume`, **plus advanced technical indicators** (`MA20`, `MA50`, `MACD`, `RSI`, `UpperBB`, `LowerBB`, `ATR`), and **time-based features** (`DayOfWeek`, `DayOfYear`).
    * **Architecture**: Utilizes **stacked LSTM layers** and a **Bidirectional LSTM** for deeper pattern recognition. Includes `Dropout` layers for regularization.
    * **Goal**: Leverage comprehensive market data and a more powerful neural network architecture for superior predictions.
    * **Learning**: Complex feature engineering, advanced deep learning architectures, and robust evaluation.

### 🚀 Technologies Used

* **Python**: Core programming language.
* **Pandas**: Data loading, manipulation, and feature engineering.
* **NumPy**: Numerical operations.
* **TensorFlow/Keras**: Building and training LSTM neural networks.
* **Scikit-learn**: Data preprocessing (MinMaxScaler) and evaluation metrics.
* **Matplotlib & Seaborn**: Data visualization and plotting results.

### 📊 Dataset

The project utilizes a historical stock price dataset for **Alphabet Inc.'s Class A (GOOGL)** shares.

* **File**: `GoogleStockPrices.csv`
* **Timeframe**: January 1, 2015 - December 31, 2024 (nearly a decade of daily data).
* **Columns**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
* **Source**: Derived from Google Finance (trading days only).

### 📁 Project Structure

```bash
├── Dataset                       # The dataset and data visualisation used for the project
└── Models                        # Main Python scripts containing the differnet LSTM model
└── README.md                     # This README file
```
### 💡 Key Learnings & Insights

* **Impact of Feature Engineering**: Demonstrated how incorporating diverse features (technical indicators, time-based data) can enrich the model's understanding of market dynamics.
* **Deep Learning Architectures**: Explored the benefits of stacked and bidirectional LSTMs for capturing complex temporal dependencies.
* **Regularization**: Applied Dropout to mitigate overfitting in more complex models.
* **Evaluation Metrics**: Utilized a comprehensive suite of regression metrics (MSE, RMSE, MAE, R2, MAPE) for robust model assessment.
* **Data vs. Complexity Trade-off**: Highlighted that increasing model complexity requires sufficient and diverse data to avoid performance degradation (e.g., due to overfitting or insufficient signal in new features). This was evident when the "Advanced" model initially underperformed the "Multi-Feature" model, suggesting tuning or more external data might be needed.

### 🔮 Future Enhancements

* **Hyperparameter Optimization**: Implement automated tuning (e.g., using Keras Tuner) to systematically find the best model configuration.
* **External Data Integration**: Incorporate news sentiment, macroeconomic indicators, or fundamental company data to provide external context.
* **Walk-Forward Validation**: Implement a walk-forward validation strategy for a more realistic assessment of real-world performance.
* **Price Movement Classification**: Reframe the problem to predict price direction (up/down/flat) instead of exact price.
* **Ensemble Methods**: Combine predictions from multiple models (e.g., LSTMs with ARIMA or other ML models).
* **Deployment**: Build a simple web application to deploy the trained model for interactive predictions.

