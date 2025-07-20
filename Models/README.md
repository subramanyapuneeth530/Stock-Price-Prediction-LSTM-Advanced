### 1. `basic_lstm_model.py`


* **Explanation**: This Python script implements a fundamental Long Short-Term Memory (LSTM) neural network for univariate time series forecasting. Its primary objective is to predict the next day's closing price of GOOGL stock by exclusively utilizing **only the past `Close` prices** as its input feature.

    The script follows a standard machine learning pipeline:
    1.  **Data Loading and Preprocessing**: It loads the `GoogleStockPrices.csv` file, converts dates, and sorts the data. Crucially, it extracts only the 'Close' price column and then **scales** this data using `MinMaxScaler` to a range between 0 and 1, which is essential for neural network training efficiency. It then creates **sequences** (e.g., using the past 60 days' closing prices to predict the 61st day's price) and splits the data into training (80%) and testing (20%) sets, ensuring the temporal order is preserved.
    2.  **Model Building**: A simple `Sequential` Keras model is constructed with a single `LSTM` layer and a `Dense` output layer. The LSTM is configured to accept sequences of 'Close' prices. The model is compiled with the 'adam' optimizer and 'mean_squared_error' as the loss function, suitable for regression tasks.
    3.  **Model Training**: The model is trained on the prepared training data. `EarlyStopping` and `ReduceLROnPlateau` callbacks are used to prevent overfitting and optimize the learning process by monitoring the validation loss.
    4.  **Evaluation and Prediction**: After training, the model makes predictions on the unseen test set. These predictions, along with the actual test prices, are **inverse-transformed** back to their original dollar scale. Key regression metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R2), and Mean Absolute Percentage Error (MAPE) are calculated and printed to quantitatively assess the model's performance.
    5.  **Visualization**: The script generates two plots: one showing the training and validation loss over epochs (to monitor convergence and overfitting), and another comparing the actual GOOGL stock prices against the model's predicted prices on the test set.

    This model serves as a robust baseline, demonstrating the core concepts of LSTM-based time series prediction and data preparation for univariate data.

* **Key Learnings**:
    * Understanding the basic architecture of an LSTM for time series.
    * Essential data preprocessing steps for sequential data (scaling, sequence creation).
    * The importance of train-test splitting for time series (preserving temporal order).
    * Using callbacks (`EarlyStopping`, `ReduceLROnPlateau`) to manage the training process.
    * Interpreting common regression evaluation metrics (MSE, RMSE, MAE, R2, MAPE).

* **Pros**:
    * Simple to understand and implement, making it an excellent starting point for LSTM beginners.
    * Provides a clear baseline performance for comparison with more complex models.
    * Effectively captures strong autocorrelation in time series data.

* **Cons**:
    * Ignores all other potentially relevant features (Open, High, Low, Volume, etc.), limiting its ability to capture complex market dynamics.
    * May underperform significantly in volatile or rapidly changing market conditions as it lacks broader context.
    * Its predictions are solely based on historical closing price patterns, without considering external factors.

* **Results**:
    ```
    --- Model Performance Metrics ---
    Mean Squared Error (MSE): 24.12
    Root Mean Squared Error (RMSE): 4.91
    Mean Absolute Error (MAE): 3.81
    R-squared (R2): 0.9662
    Mean Absolute Percentage Error (MAPE): 2.65%
    ```
    ![Figure_1](https://github.com/user-attachments/assets/9d6d70e4-68a2-47ec-9c3b-52cc998db17d)
    ![Figure_2](https://github.com/user-attachments/assets/ca1c6b36-eed0-4a43-9232-dec12a449475)

    **Analysis of Results:**

    The performance metrics for the Basic LSTM model are **exceptionally strong** 💪, especially considering it only utilizes past closing prices for prediction.

    * **Low Absolute Errors (RMSE: $4.91, MAE: $3.81)**: The model's predictions are, on average, within approximately \$4 to \$5 of the actual GOOGL stock price. For a stock often priced in the hundreds, this indicates a remarkably **high level of precision**. The close proximity of MAE and RMSE suggests that there aren't many significantly large individual errors skewing the results.

    * **High R-squared (R2): 0.9662**: This is an **outstanding** R2 score. It means that about **96.62% of the variation** in GOOGL's closing prices on the test set can be explained by your model. This demonstrates a **very strong fit** and indicates the model effectively captures the underlying price movements.

    * **Low Mean Absolute Percentage Error (MAPE): 2.65%**: A MAPE of just **2.65%** is excellent for financial forecasting. It implies that, on average, the prediction error is only about 2.65% of the actual stock price value, making the model's accuracy **highly interpretable and practically valuable**.

    **Visual Inference from Plots:**
  * **Loss Plot (Figure 2: Model Loss Over Epochs)**:
        * This plot shows the **training loss (blue line)** and **validation loss (orange line)** decreasing over epochs.
        * Both loss curves show a consistent downward trend, indicating that the model is **learning effectively** from the data.
        * The validation loss closely follows the training loss and eventually **plateaus or shows minimal increase**, suggesting that the model is **not significantly overfitting** to the training data. The `EarlyStopping` callback likely intervened when the validation loss stopped improving, ensuring the model generalizes well to unseen data. This stable loss behavior contributes to the reliable performance observed in the prediction plot and metrics.
  * **Prediction Plot (Figure 1: Actual vs. Predicted Prices)**:
        * The plot visually confirms the **strong performance** indicated by the metrics. The **predicted price line (blue) closely tracks the actual price line (red)**.
        * The model effectively captures the **overall trend** of GOOGL stock, as well as many of the smaller fluctuations and turns.
        * While there might be slight lags or minor deviations during sharp upswings or downswings, the **general fit is excellent**. This visual alignment strongly supports the high R2 score.

* **Conclusion**:

  This Basic LSTM model serves as a **highly effective baseline** for GOOGL stock price prediction. Its strong performance, both quantitatively through metrics and qualitatively through visual plots, primarily driven by successfully leveraging the inherent autocorrelation in stock prices, underscores the power of LSTMs even with a limited feature set. While more complex models might aim for marginal gains or address specific market conditions, this initial model sets a very high standard for accuracy.

### 2. `multi_feature_lstm_model.py`

* **Project Title**: Multi-Feature LSTM Model
* **Explanation**: This Python script enhances the basic LSTM model by incorporating **multiple input features** to predict the next day's closing price of GOOGL stock. Unlike the univariate baseline, this model leverages a richer set of daily market data (`Open`, `High`, `Low`, `Close`, `Volume`), aiming for improved accuracy and a more comprehensive understanding of market dynamics.

    The script follows an extended machine learning pipeline:

    1.  **Data Loading and Multi-Feature Preprocessing**: It loads the `GoogleStockPrices.csv` file, converts dates, and sorts the data. It selects multiple features (`Open`, `High`, `Low`, `Close`, `Volume`), which are then **scaled together** using `MinMaxScaler` to a range between 0 and 1. This collective scaling is crucial for neural networks when dealing with diverse input scales. It creates **sequences** (e.g., using the past 60 days of all selected features) to predict the 'Close' price for the next day. The data is then split into training (80%) and testing (20%) sets, preserving temporal order.
    2.  **Model Building**: A `Sequential` Keras model is constructed with an **LSTM layer** configured to accept multi-feature input sequences. A **`Dropout` layer** is explicitly added after the LSTM as a regularization technique to **prevent overfitting** by randomly setting a fraction of input units to 0 during training. A `Dense` output layer produces the single predicted closing price. The model is compiled with the 'adam' optimizer and 'mean_squared_error' loss.
    3.  **Model Training**: The model is trained on the prepared training data. `EarlyStopping` and `ReduceLROnPlateau` callbacks are utilized to monitor validation loss, preventing overfitting by stopping training if performance plateaus and adjusting the learning rate dynamically.
    4.  **Evaluation and Prediction**: Predictions are made on the unseen test set. These scaled predictions are **inverse-transformed** back to their original dollar scale using the `MinMaxScaler` fitted on all features. Standard regression metrics (MSE, RMSE, MAE, R2, MAPE) are calculated to quantitatively assess the model's performance.
    5.  **Visualization**: The script generates plots showing the training and validation loss over epochs and a comparison of actual vs. predicted GOOGL stock prices on the test set.

    This model signifies a substantial step forward from the basic baseline, demonstrating how incorporating a richer feature set can enhance the predictive power of an LSTM model for financial time series.

* **Key Learnings**:
    * **Handling Multivariate Time Series**: How to structure input data for LSTMs when using multiple features.
    * **Comprehensive Feature Scaling**: The necessity of scaling all input features consistently.
    * **Regularization with Dropout**: Its importance in mitigating overfitting when models become more complex.
    * **Improved Contextual Learning**: How providing more market data (Open, High, Low, Volume) allows the model to learn richer patterns.

* **Pros**:
    * **Captures More Market Context**: By using OHLCV data, it gets a more holistic view of daily market activity compared to just closing prices.
    * Aims for **better predictive performance** than the basic model by leveraging more relevant information.
    * Introduces **Dropout**, a crucial technique for building more robust deep learning models.

* **Cons**:
    * **Requires Careful Feature Scaling**: All input features must be scaled correctly.
    * **Sensitive to Overfitting**: Despite dropout, increasing complexity still requires careful tuning to avoid overfitting, especially if the dataset's diversity is limited.
    * Still relies solely on internal stock data, **lacking external drivers** like news sentiment or macroeconomic factors.

* **Results**:

    ```
    --- Model Performance Metrics (Multi-Feature LSTM) ---
    Mean Squared Error (MSE): 35.49
    Root Mean Squared Error (RMSE): 5.96
    Mean Absolute Error (MAE): 4.73
    R-squared (R2): 0.9503
    Mean Absolute Percentage Error (MAPE): 3.45%
    ```
    <img width="1920" height="1015" alt="Figure_3" src="https://github.com/user-attachments/assets/7d9b507d-b0ff-4f8a-acdf-995916c03e7e" />
    <img width="1920" height="1015" alt="Figure_4" src="https://github.com/user-attachments/assets/2d7c6473-97dd-4772-9cf3-171d9c444d44" />




    **Analysis of Results:**

    The performance metrics for the Multi-Feature LSTM model demonstrate **strong predictive capabilities** 💪, successfully leveraging the expanded set of input features.

    * **Error Metrics (RMSE: $5.96, MAE: $4.73)**: The average prediction error is approximately \$5.96 (RMSE) and the average absolute error is \$4.73 (MAE). While these are slightly higher than the basic LSTM's impressive results (Basic LSTM RMSE: \$4.91, MAE: \$3.81), they still represent a **very good level of precision** for stock price prediction, especially given the increased complexity of the model and the challenge of forecasting in financial markets.
    * **R-squared (R2): 0.9503**: An R2 score of **0.9503** indicates that approximately **95.03% of the variance** in GOOGL's closing prices on the test set can be explained by this model. This signifies a **very strong fit** 👍, showing that the model effectively captures the underlying price movements using the multiple input features.
    * **Mean Absolute Percentage Error (MAPE): 3.45%**: A MAPE of **3.45%** signifies that, on average, the predictions deviate by only about 3.45% from the actual price. This is a **highly acceptable percentage error** for stock forecasting, indicating good proportional accuracy.

    **Visual Inference from Plots:**

    * **Prediction Plot (Figure 3: Actual vs. Predicted Prices)**:
        * The plot visually confirms that the **predicted price line (blue) closely tracks the actual price line (red)**.
        * The model effectively captures the **overall trend** and many of the daily fluctuations of GOOGL stock.
        * There may be instances of minor lag or slightly less precise capture during very sharp price changes compared to the actual movements, but the general alignment remains strong. This visual coherence supports the high R2 score.

    * **Loss Plot (Figure 4: Multi-Feature Model Loss Over Epochs)**:
        * This plot displays the **training loss (blue line)** and **validation loss (orange line)** decreasing over epochs.
        * Both loss curves show a consistent downward trend, confirming that the model is **learning effectively** from the data.
        * The validation loss tracks the training loss well and eventually plateaus or shows a minimal increase, suggesting that the model is **generalizing adequately** and that regularization (dropout, early stopping) is effectively preventing significant overfitting.

* **Conclusion**:

    The Multi-Feature LSTM model successfully integrates additional market data (Open, High, Low, Close, Volume) to predict GOOGL stock prices. While its absolute error metrics are slightly higher than the Basic LSTM, it still demonstrates **very strong predictive power** with an R2 of over 0.95 and a MAPE below 3.5%. This indicates that the added features are indeed providing valuable context, even if the basic model's simplicity sometimes benefits from directly capitalizing on strong autocorrelation. This model represents a significant and successful step in building a more comprehensive and robust stock prediction system. The slight increase in errors compared to the Basic LSTM could indicate that the added complexity requires even **finer hyperparameter tuning** 🛠️ or the incorporation of **external data** 🌐 for the model to truly outshine its simpler counterpart.

### 3. `advanced_multi_feature_lstm_model.py`

* **Project Title**: Advanced Multi-Feature LSTM Model
* **Explanation**: This Python script represents the most sophisticated iteration in our stock price prediction series. It significantly enhances the multi-feature LSTM model by incorporating a **wider array of inputs** and employing a **more complex and powerful deep learning architecture**. The primary goal is to achieve superior predictive performance for GOOGL stock by leveraging extensive market information.

    The script's refined pipeline includes:

    1.  **Advanced Data Loading, Preprocessing, and Feature Engineering**:
        * It loads the `GoogleStockPrices.csv` file and processes dates.
        * Beyond just OHLCV, it calculates and adds several **advanced technical indicators**:
            * **Moving Averages** (`MA20`, `MA50`): For trend identification.
            * **MACD (Moving Average Convergence Divergence)**: For trend following and momentum.
            * **RSI (Relative Strength Index)**: A momentum oscillator for overbought/oversold conditions.
            * **Bollinger Bands** (`UpperBB`, `LowerBB`): For volatility and price envelopes.
            * **ATR (Average True Range)**: For measuring market volatility.
        * It also extracts **time-based features** (`DayOfWeek`, `DayOfYear`) to capture potential cyclical patterns.
        * Rows containing `NaN` values resulting from the calculation of these rolling-window indicators are **dropped** to ensure clean data.
        * All selected features are then **scaled** using `MinMaxScaler`.
        * Finally, sequences are created from this rich multivariate data to serve as input for the LSTM, and the data is split into training (80%) and testing (20%) sets, maintaining temporal order.
    2.  **Advanced Model Building**: A `Sequential` Keras model is constructed with a significantly enhanced architecture:
        * **Stacked LSTM Layers**: It utilizes **multiple LSTM layers** (configured via `lstm_units` as a list, e.g., `[100, 100]`). This allows the model to learn more complex and hierarchical representations of the time series data. `return_sequences=True` is used for intermediate LSTM layers to pass the full sequence output to the next layer.
        * **Bidirectional LSTM**: The **first LSTM layer is wrapped in `Bidirectional`**. This enables the network to process input sequences in both a forward and backward direction, potentially capturing dependencies and context that a unidirectional LSTM might miss.
        * **Dropout Layers**: `Dropout` layers are strategically placed after each LSTM layer. This is a crucial **regularization technique** that randomly deactivates a fraction of neurons during training, significantly helping to **combat overfitting** in this more complex model.
        * A `Dense` output layer produces the single predicted closing price. The model is compiled with the 'adam' optimizer and 'mean_squared_error' loss.
    3.  **Model Training**: The model is trained on the prepared training data. The training process uses more **patient callbacks** for `EarlyStopping` and `ReduceLROnPlateau` (increased patience values and a smaller `min_lr`) to allow the more complex model sufficient time to converge while preventing overfitting.
    4.  **Evaluation and Prediction**: Predictions are generated on the unseen test set and inverse-transformed back to original scale. Comprehensive regression metrics (MSE, RMSE, MAE, R2, MAPE) are calculated to quantify the model's performance.
    5.  **Visualization**: Plots showing training/validation loss over epochs and actual vs. predicted GOOGL stock prices are generated.

    This model represents the pinnacle of complexity in this project series, aiming to capture the most intricate patterns in stock prices using a powerful deep learning architecture and extensive feature engineering.

* **Key Learnings**:
    * **Comprehensive Feature Engineering**: The process of selecting, calculating, and integrating a wide array of financial and time-based indicators.
    * **Advanced Deep Learning Architectures**: Implementing and understanding the benefits of **stacked LSTMs** and **Bidirectional LSTMs** for complex sequence modeling.
    * **Robust Regularization**: The critical role of `Dropout` and finely-tuned callbacks in managing overfitting in high-capacity models.
    * **Complexity vs. Performance Trade-offs**: Understanding that increased model complexity demands careful tuning and potentially larger, more diverse datasets to achieve optimal generalization.

* **Pros**:
    * **Leverages Extensive Market Insights**: Utilizes a broad range of technical and temporal features, providing the model with much richer context than simpler versions.
    * **Advanced Pattern Recognition**: The deeper, bidirectional architecture is theoretically capable of capturing highly intricate and long-range dependencies in the data.
    * Designed for **potentially superior performance** and robustness in varying market conditions due to its comprehensive input and powerful learning capabilities.

* **Cons**:
    * **Significantly Higher Computational Cost**: Training and prediction are more resource-intensive due to the larger number of features and model parameters.
    * **Highly Sensitive to Hyperparameter Tuning**: Requires very meticulous tuning of `lstm_units`, `dropout_rate`, learning rate schedules, and sequence length to achieve optimal performance and avoid overfitting.
    * **Increased Risk of Overfitting**: Despite regularization, the high capacity of the model can lead to overfitting if the dataset is not sufficiently large, diverse, or clean, potentially causing performance degradation compared to simpler models (as observed in some initial runs).
    * Still relies primarily on historical price-derived data, **lacking external drivers** like news sentiment, macroeconomic factors, or fundamental company data, which can cause sudden, unpredictable shifts.

* **Results**:
    ```
    --- Model Performance Metrics (Advanced Multi-Feature LSTM) ---
    Mean Squared Error (MSE): 52.02
    Root Mean Squared Error (RMSE): 7.21
    Mean Absolute Error (MAE): 5.82
    R-squared (R2): 0.9239
    Mean Absolute Percentage Error (MAPE): 3.97%
    ```
    <img width="1920" height="1015" alt="Figure_5" src="https://github.com/user-attachments/assets/716ed4de-6d9a-407f-8363-29b845fff138" />
    <img width="1920" height="1015" alt="Figure_6" src="https://github.com/user-attachments/assets/1d1ea8e8-abd8-4b55-b8b5-c99bc03c02b8" />


    **Analysis of Results:**

    The performance metrics for the Advanced Multi-Feature LSTM model indicate that while the model still captures the stock's overall trend, its **predictive accuracy has slightly decreased** compared to the previous Multi-Feature LSTM model.

    * **Error Metrics (RMSE: $7.21, MAE: $5.82)**: The average prediction error (RMSE) is approximately \$7.21, and the average absolute error (MAE) is \$5.82. These values are higher than those achieved by the Multi-Feature LSTM (RMSE: \$5.96, MAE: \$4.73), suggesting a **slight reduction in absolute prediction precision**.
    * **R-squared (R2): 0.9239**: An R2 score of **0.9239** is still strong, indicating that the model explains about **92.39% of the variance** in GOOGL's closing prices. However, this is a decrease from the previous Multi-Feature LSTM's R2 of 0.9503, implying a **marginally weaker fit** to the data.
    * **Mean Absolute Percentage Error (MAPE): 3.97%**: A MAPE of **3.97%** is acceptable for financial forecasting, but it's higher than the previous model's 3.45%, showing a **marginal increase in proportional error**.

    **Visual Inference from Plots:**

    * **Prediction Plot (Figure 5: Actual vs. Predicted Prices)**:
        * The plot shows the **predicted price line (blue) generally following the actual price line (red)**, indicating that the model successfully captures the overall trend of GOOGL stock.
        * However, compared to the simpler models, there appear to be **more noticeable deviations, especially during periods of high volatility or sharp price changes**. The predicted line might exhibit increased lag or undershooting/overshooting of significant movements. This visual observation aligns with the slight increase in error metrics.

    * **Loss Plot (Figure 6: Advanced Multi-Feature Model Loss Over Epochs)**:
        * This plot displays the **training loss (blue line)** and **validation loss (orange line)** decreasing over epochs.
        * The loss curves show the model is **learning during training**. However, if the validation loss shows a more pronounced divergence from the training loss (e.g., significantly flattening out or starting to rise while training loss continues to decrease), it would strongly suggest **overfitting**. Without seeing the exact scale and behavior relative to the training loss, the current metrics suggest that while learning occurred, the generalization might not be as robust as expected from the added complexity.

* **Conclusion**:

    The Advanced Multi-Feature LSTM model, despite its increased complexity and expanded feature set, has **not yet surpassed the performance** of the Multi-Feature LSTM model in its current configuration. The slight increase in all error metrics and a marginal decrease in R2 suggest that the added model capacity and new features are **not being optimally leveraged**, and may even be contributing to a higher degree of overfitting or noise in the predictions.

    This outcome highlights a crucial lesson in machine learning: **more complex models require meticulous hyperparameter tuning** 🛠️ and potentially **larger, more diverse datasets** 🌐 to fully realize their potential and avoid pitfalls like overfitting. The model's inherent power remains, but unlocking it requires further optimization and validation.
