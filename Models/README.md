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
