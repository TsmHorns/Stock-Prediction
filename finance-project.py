
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Download Historical Stock Data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Add Technical Indicators and Prepare the Data
def add_technical_indicators(stock_data):
    stock_data['MA_10'] = stock_data['Adj Close'].rolling(window=10).mean()
    stock_data['MA_50'] = stock_data['Adj Close'].rolling(window=50).mean()
    stock_data['RSI'] = calculate_rsi(stock_data['Adj Close'])
    stock_data['Volatility'] = stock_data['Adj Close'].rolling(window=10).std()
    stock_data['Target'] = stock_data['Adj Close'].shift(-1)
    stock_data = stock_data.dropna()
    return stock_data

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(stock_data):
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50', 'RSI', 'Volatility']]
    y = stock_data['Target']
    return X, y

# Download data for Nvidia (NVDA) from 2010 to 2024
ticker = "NVDA"
start_date = "2010-01-01"
end_date = "2024-09-05"
stock_data = download_stock_data(ticker, start_date, end_date)

# Add technical indicators
stock_data = add_technical_indicators(stock_data)

# Prepare the data
X, y = prepare_data(stock_data)

# Step 3: Preprocess the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Define the B-Spline Activation Function
def b_spline_activation(x, degree=3, knots=None, max_iterations=10):
    if knots is None:
        knots = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), degree + 2)

    def bspline_basis(x, degree, knots, i):
        if degree == 0:
            return tf.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)
        else:
            left_num = x - knots[i]
            left_denom = knots[i + degree] - knots[i] + 1e-6
            left = left_num / left_denom

            right_num = knots[i + degree + 1] - x
            right_denom = knots[i + degree + 1] - knots[i + 1] + 1e-6
            right = right_num / right_denom

            return left * bspline_basis(x, degree - 1, knots, i) + right * bspline_basis(x, degree - 1, knots, i + 1)

    n_basis = tf.shape(knots)[0] - degree - 1
    activation = tf.zeros_like(x)
    i = tf.constant(0)

    while_condition = lambda i, activation: tf.logical_and(tf.less(i, n_basis), tf.less(i, max_iterations))

    def body(i, activation):
        activation += bspline_basis(x, degree, knots, i)
        return tf.add(i, 1), activation

    _, activation = tf.while_loop(while_condition, body, [i, activation], maximum_iterations=max_iterations)

    return activation

# Step 5: Create the Enhanced FastKAN Model
def create_fastkan_model(input_dim, degree=3):
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # Hidden layers with custom B-spline activation
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Lambda(lambda x: b_spline_activation(x, degree=degree)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Lambda(lambda x: b_spline_activation(x, degree=degree)))
    model.add(layers.Dropout(0.2))

    # Output layer
    model.add(layers.Dense(1))

    # Compile the model with a smaller learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Step 6: Train and Evaluate the Model
fastkan_model = create_fastkan_model(X_train.shape[1], degree=3)

history = fastkan_model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

test_loss, test_mae = fastkan_model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")

# Step 7: Visualize Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

# Plot Actual vs Predicted Values
plt.subplot(1, 2, 2)
y_pred = fastkan_model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')

plt.tight_layout()
plt.show()

def plot_stock_prices(stock_data, y_test, y_pred, start_date, end_date):
    # Ensure stock_data has the correct DateTime index
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        raise ValueError("stock_data must have a DateTime index")

    # Create a DataFrame for predictions with the same index as stock_data
    pred_df = pd.DataFrame(index=stock_data.index, data={'Predicted': np.nan})

    # Align predictions with test dates
    test_dates = stock_data.index[-len(y_test):]
    pred_df.loc[test_dates, 'Predicted'] = y_pred.flatten()

    # Fill missing predictions with NaNs (for the actual stock data range)
    stock_data = stock_data.copy()
    stock_data['Predicted'] = pred_df['Predicted']

    # Plotting
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(stock_data.index, stock_data['Adj Close'], label='Actual')
    plt.title('Actual Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    plt.subplot(1, 2, 2)
    plt.plot(stock_data.index, stock_data['Predicted'], label='Predicted', color='orange')
    plt.title('Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    plt.tight_layout()
    plt.show()

# Call the function to plot with the date range
plot_stock_prices(stock_data, y_test, y_pred, start_date, end_date)
