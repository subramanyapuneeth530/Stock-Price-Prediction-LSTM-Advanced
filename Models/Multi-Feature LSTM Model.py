import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Configuration ---
CONFIG = {
    'file_path': 'GoogleStockPrices.csv',
    'features': ['Open', 'High', 'Low', 'Close', 'Volume'], # Multi-features now!
    'target_column': 'Close', # Still predicting Close price
    'sequence_length': 60,  # Number of past days to consider for prediction
    'train_split_ratio': 0.8,
    'lstm_units': 100, # Increased units for more complexity
    'dropout_rate': 0.2, # Added dropout for regularization
    'epochs': 100, # Increased epochs
    'batch_size': 32,
    'patience_early_stopping': 15, # Increased patience
    'patience_reduce_lr': 7,
    'min_lr': 0.0001,
    'plot_results': True,
    'company_name': 'GOOGL',
}

# --- 1. Load and Preprocess Data ---
def load_and_prepare_multi_feature_data(file_path, features, target_column, sequence_length, train_split_ratio):
    """
    Loads stock data, selects multiple features, scales them,
    creates sequences, and splits into train/test sets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Select the features and convert to numpy array
    data = df[features].values

    # Scale the data using MinMaxScaler
    # Fit scaler on ALL features
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

    # X is already in 3D shape (samples, timesteps, features) because of scaled_data[:, :]
    # y is 2D (samples, 1) after np.array(y) but will become 1D with y.reshape(-1,) in training for simplicity

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_split_ratio), shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler, df, target_col_idx

# --- 2. Build the Multi-Feature LSTM Model ---
def build_multi_feature_lstm_model(sequence_length, num_features, lstm_units, dropout_rate):
    """
    Builds a Sequential Keras model with an LSTM layer for multi-feature input.
    Adds Dropout for regularization.
    """
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=False, input_shape=(sequence_length, num_features)),
        Dropout(dropout_rate), # Add dropout layer
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
def evaluate_and_predict(model, X_test, y_test, scaler, target_col_idx, num_features):
    """
    Makes predictions on the test set and inverse transforms them to original scale.
    Calculates and prints multiple evaluation metrics.
    """
    predictions_scaled = model.predict(X_test)

    # To inverse transform predictions, we need to create a dummy array
    # with the same number of features as the scaler was fitted on.
    # Fill the target column with predictions and other columns with zeros (or any dummy value).
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
    # Handle potential division by zero for MAPE if actual_prices contains zeros
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    mape = np.nan_to_num(mape, nan=np.inf) # Replace NaN with inf, if any
    
    print("\n--- Model Performance Metrics (Multi-Feature LSTM) ---")
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
    # `y` is created from `scaled_data` starting from `sequence_length` index.
    # `y_test` is the last `(1 - train_split_ratio)` portion of `y`.
    
    # Calculate the number of data points effectively used for sequence creation
    total_effective_data_points = len(original_df) - sequence_length
    
    # Calculate the index where the test set starts in the 'effective' data
    train_effective_data_points = int(total_effective_data_points * train_split_ratio)
    
    # The start index in the original dataframe for the test set's corresponding dates
    # It's the sequence_length offset + the number of training points (from the effective set)
    start_test_date_index = sequence_length + train_effective_data_points
    
    # Slice the dates from the original_df to match the length of actual_prices/predicted_prices
    plot_dates = original_df['Date'].iloc[start_test_date_index : start_test_date_index + len(actual_prices)]
    
    plt.plot(plot_dates, actual_prices, color='red', label=f'Actual {company_name} Price')
    plt.plot(plot_dates, predicted_prices, color='blue', label=f'Predicted {company_name} Price')
    plt.title(f'{company_name} Stock Price Prediction (Multi-Feature LSTM)', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Load, prepare, and split data for multi-feature input
        X_train, X_test, y_train, y_test, scaler, original_df, target_col_idx = load_and_prepare_multi_feature_data(
            CONFIG['file_path'], CONFIG['features'], CONFIG['target_column'],
            CONFIG['sequence_length'], CONFIG['train_split_ratio']
        )

        # Build the Multi-Feature LSTM model
        # The number of features for input_shape is now len(CONFIG['features'])
        num_features = len(CONFIG['features'])
        model = build_multi_feature_lstm_model(
            CONFIG['sequence_length'], num_features, CONFIG['lstm_units'], CONFIG['dropout_rate']
        )
        model.summary()

        # Train the model
        print("\n--- Training Multi-Feature Model ---")
        history = train_model(
            model, X_train, y_train,
            CONFIG['epochs'], CONFIG['batch_size'],
            CONFIG['patience_early_stopping'], CONFIG['patience_reduce_lr'], CONFIG['min_lr']
        )

        # Evaluate and make predictions
        print("\n--- Evaluating Multi-Feature Model ---")
        predicted_prices, actual_prices = evaluate_and_predict(
            model, X_test, y_test, scaler, target_col_idx, num_features
        )

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Multi-Feature Model Loss Over Epochs')
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
