import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Configuration ---
CONFIG = {
    'file_path': 'GoogleStockPrices.csv',
    # New: More features including technical indicators and time-based features
    'features': ['Open', 'High', 'Low', 'Close', 'Volume',
                 'MA20', 'MA50', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'ATR', # Technical indicators
                 'DayOfWeek', 'DayOfYear'], # Time-based features
    'target_column': 'Close', # Still predicting Close price
    'sequence_length': 60,  # Number of past days to consider for prediction
    'train_split_ratio': 0.8,
    'lstm_units': [100, 100], # Stacked LSTM layers with 100 units each
    'dropout_rate': 0.3, # Increased dropout for more complex model
    'epochs': 200, # Increased epochs to allow more training
    'batch_size': 32,
    'patience_early_stopping': 20, # Increased patience
    'patience_reduce_lr': 10,
    'min_lr': 0.00001, # Smaller min learning rate
    'plot_results': True,
    'company_name': 'GOOGL',
}

# --- 1. Load, Preprocess Data & Calculate Advanced Features ---
def load_and_prepare_advanced_data(file_path, features, target_column, sequence_length, train_split_ratio):
    """
    Loads stock data, calculates advanced technical and time-based features,
    scales them, creates sequences, and splits into train/test sets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # --- Calculate Technical Indicators ---
    # Moving Averages (Already in original code, but explicitly add here for consistency)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # MACD (Moving Average Convergence Divergence)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean() # Signal line also often used, but we'll just add MACD for simplicity here

    # RSI (Relative Strength Index)
    window_rsi = 14
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window_rsi, min_periods=window_rsi).mean()
    avg_loss = loss.rolling(window=window_rsi, min_periods=window_rsi).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].replace([np.inf, -np.inf], 100).fillna(50) # Handle NaN/Inf from early periods or zero loss

    # Bollinger Bands
    window_bb = 20
    df['BB_Mid'] = df['Close'].rolling(window=window_bb).mean()
    df['BB_Std'] = df['Close'].rolling(window=window_bb).std()
    df['UpperBB'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['LowerBB'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    # ATR (Average True Range)
    window_atr = 14
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift(1))
    low_prev_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.DataFrame({'HL': high_low, 'HPC': high_prev_close, 'LPC': low_prev_close}).max(axis=1)
    df['ATR'] = tr.rolling(window=window_atr, min_periods=window_atr).mean()

    # --- Add Time-Based Features ---
    df['DayOfWeek'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['DayOfYear'] = df['Date'].dt.dayofyear # 1 to 366

    # Drop rows with NaN values created by rolling windows (e.g., first 50 or 200 days for MAs)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Select the final features for the model
    data = df[features].values

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Find the index of the target column in the 'features' list
    target_col_idx = features.index(target_column)

    # Create sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :]) # All features in X
        y.append(scaled_data[i, target_col_idx])      # Only the target column in y

    X, y = np.array(X), np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_split_ratio), shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler, df, target_col_idx

# --- 2. Build the Multi-Feature Stacked & Bidirectional LSTM Model ---
def build_advanced_lstm_model(sequence_length, num_features, lstm_units, dropout_rate):
    """
    Builds a Sequential Keras model with stacked and potentially Bidirectional LSTM layers.
    """
    model = Sequential()
    
    # First LSTM layer (Bidirectional)
    # return_sequences=True is needed for stacking LSTMs
    model.add(Bidirectional(LSTM(units=lstm_units[0], return_sequences=True), 
                            input_shape=(sequence_length, num_features)))
    model.add(Dropout(dropout_rate))

    # Second LSTM layer (can be simple LSTM if not the last)
    # return_sequences=False for the last LSTM layer before Dense output
    model.add(LSTM(units=lstm_units[1], return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 3. Train the Model ---
def train_model(model, X_train, y_train, epochs, batch_size, patience_es, patience_lr, min_lr):
    """
    Trains the LSTM model with Early Stopping and ReduceLROnPlateau callbacks.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience_es,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use a portion of training data for validation
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return history

# --- 4. Evaluate and Predict ---
def evaluate_and_predict(model, X_test, y_test, scaler, target_col_idx, num_features):
    """
    Makes predictions on the test set and inverse transforms them to original scale.
    Calculates and prints multiple evaluation metrics.
    """
    predictions_scaled = model.predict(X_test)

    # To inverse transform predictions, we need to create a dummy array
    # with the same number of features as the scaler was fitted on.
    dummy_array_for_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_array_for_predictions[:, target_col_idx] = predictions_scaled.flatten()
    predicted_prices = scaler.inverse_transform(dummy_array_for_predictions)[:, target_col_idx]

    # Similarly for actual prices (y_test was only the scaled target, need to recreate for inverse transform)
    dummy_array_for_actuals = np.zeros((len(y_test), num_features))
    dummy_array_for_actuals[:, target_col_idx] = y_test.flatten()
    actual_prices = scaler.inverse_transform(dummy_array_for_actuals)[:, target_col_idx]

    # Calculate evaluation metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    # Calculate MAPE, handle division by zero or NaN values appropriately
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices[actual_prices != 0])) * 100
    mape = np.nan_to_num(mape, nan=np.inf) # Replace NaN with inf, if any

    print("\n--- Model Performance Metrics (Advanced Multi-Feature LSTM) ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return predicted_prices, actual_prices

# --- 5. Plot Results ---
def plot_predictions(original_df, actual_prices, predicted_prices, sequence_length, train_split_ratio, company_name):
    """
    Plots the actual vs. predicted stock prices.
    """
    plt.figure(figsize=(16, 8))

    # Correct date slicing:
    total_effective_data_points = len(original_df) - sequence_length
    train_effective_data_points = int(total_effective_data_points * train_split_ratio)
    start_test_date_index = sequence_length + train_effective_data_points
    
    plot_dates = original_df['Date'].iloc[start_test_date_index : start_test_date_index + len(actual_prices)]
    
    plt.plot(plot_dates, actual_prices, color='red', label=f'Actual {company_name} Price')
    plt.plot(plot_dates, predicted_prices, color='blue', label=f'Predicted {company_name} Price')
    plt.title(f'{company_name} Stock Price Prediction (Advanced Multi-Feature LSTM)', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Load, prepare, and split data for multi-feature input with advanced features
        X_train, X_test, y_train, y_test, scaler, original_df, target_col_idx = load_and_prepare_advanced_data(
            CONFIG['file_path'], CONFIG['features'], CONFIG['target_column'],
            CONFIG['sequence_length'], CONFIG['train_split_ratio']
        )

        # Build the Advanced Multi-Feature LSTM model
        num_features = X_train.shape[2] # Get number of features from X_train shape
        model = build_advanced_lstm_model(
            CONFIG['sequence_length'], num_features, CONFIG['lstm_units'], CONFIG['dropout_rate']
        )
        model.summary()

        # Train the model
        print("\n--- Training Advanced Multi-Feature Model ---")
        history = train_model(
            model, X_train, y_train,
            CONFIG['epochs'], CONFIG['batch_size'],
            CONFIG['patience_early_stopping'], CONFIG['patience_reduce_lr'], CONFIG['min_lr']
        )

        # Evaluate and make predictions
        print("\n--- Evaluating Advanced Multi-Feature Model ---")
        predicted_prices, actual_prices = evaluate_and_predict(
            model, X_test, y_test, scaler, target_col_idx, num_features
        )

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Advanced Multi-Feature Model Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Plot predictions if enabled
        if CONFIG['plot_results']:
            plot_predictions(
                original_df,
                actual_prices,
                predicted_prices,
                CONFIG['sequence_length'],
                CONFIG['train_split_ratio'],
                CONFIG['company_name']
            )

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
