import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Configuration ---
CONFIG = {
    'file_path': 'GoogleStockPrices.csv',
    'target_column': 'Close',
    'sequence_length': 60,  # Number of past days to consider for prediction
    'train_split_ratio': 0.8,
    'lstm_units': 50,
    'epochs': 50,
    'batch_size': 32,
    'patience_early_stopping': 10,
    'patience_reduce_lr': 5,
    'min_lr': 0.0001,
    'plot_results': True,
    'company_name': 'GOOGL',
}

# --- 1. Load and Preprocess Data ---
def load_and_prepare_data(file_path, target_column, sequence_length, train_split_ratio):
    """
    Loads stock data, scales it, creates sequences, and splits into train/test sets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Use only the target column (Close price)
    data = df[[target_column]].values

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM input (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_split_ratio), shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler, df

# --- 2. Build the LSTM Model ---
def build_lstm_model(sequence_length, lstm_units):
    """
    Builds a Sequential Keras model with an LSTM layer.
    """
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=False, input_shape=(sequence_length, 1)),
        Dense(units=1)
    ])
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
def evaluate_and_predict(model, X_test, y_test, scaler):
    """
    Makes predictions on the test set and inverse transforms them to original scale.
    Calculates and prints multiple evaluation metrics.
    """
    predictions = model.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate evaluation metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    print("\n--- Model Performance Metrics ---")
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
    # The actual_prices and predicted_prices correspond to the y_test.
    # y_test is the tail of the 'y' array, which itself starts at 'sequence_length' index in scaled_data.
    # So, the dates for plotting should start from the index in original_df
    # that corresponds to the first element of y_test.
    
    # Calculate the starting index of the test set in the original dataframe
    # after accounting for the sequence_length for feature creation.
    
    # Total data points after sequence creation (length of X or y)
    total_effective_data_points = len(original_df) - sequence_length
    
    # Number of data points in the training set (from X_train/y_train)
    train_effective_data_points = int(total_effective_data_points * train_split_ratio)
    
    # Starting index in original_df for the test predictions
    # This is the point after the initial sequence_length + the training data
    start_test_date_index = sequence_length + train_effective_data_points
    
    # Ensure the plot_dates array matches the length of actual_prices/predicted_prices
    plot_dates = original_df['Date'].iloc[start_test_date_index : start_test_date_index + len(actual_prices)]
    
    # Fallback/alternative for plot_dates (simpler, should also work if data aligns well)
    # plot_dates = original_df['Date'].iloc[-len(actual_prices):]


    plt.plot(plot_dates, actual_prices, color='red', label=f'Actual {company_name} Price')
    plt.plot(plot_dates, predicted_prices, color='blue', label=f'Predicted {company_name} Price')
    plt.title(f'{company_name} Stock Price Prediction (Basic LSTM)', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Load, prepare, and split data
        X_train, X_test, y_train, y_test, scaler, original_df = load_and_prepare_data(
            CONFIG['file_path'], CONFIG['target_column'], CONFIG['sequence_length'], CONFIG['train_split_ratio']
        )

        # Build the LSTM model
        model = build_lstm_model(CONFIG['sequence_length'], CONFIG['lstm_units'])
        model.summary()

        # Train the model
        print("\n--- Training Model ---")
        history = train_model(
            model, X_train, y_train,
            CONFIG['epochs'], CONFIG['batch_size'],
            CONFIG['patience_early_stopping'], CONFIG['patience_reduce_lr'], CONFIG['min_lr']
        )

        # Evaluate and make predictions
        print("\n--- Evaluating Model ---")
        predicted_prices, actual_prices = evaluate_and_predict(model, X_test, y_test, scaler)

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
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
