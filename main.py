import yfinance as yf
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import feedparser
import datetime
from keras.models import load_model
import plotly.graph_objs as go
import plotly.subplots as sp
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import os

top_10_cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD', 'LUNA-USD']
start_date = "2010-01-01"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
global future_days_entry
top_10_cryptosrss = top_10_cryptos



def get_crypto_data(crypto, start_date, end_date):
    data = yf.download(tickers=crypto, start=start_date, end=end_date)
    return data

def preprocess_data(data, split_ratio=0.8):
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    days_to_look_back = 60

    # Split into training and testing sets
    training_data_len = int(np.ceil(len(scaled_data) * split_ratio))
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - days_to_look_back:]

    # Prepare the data for LSTM
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(days_to_look_back, len(train_data)):
        X_train.append(train_data[i - days_to_look_back:i])
        y_train.append(train_data[i])

    for i in range(days_to_look_back, len(test_data)):
        X_test.append(test_data[i - days_to_look_back:i])
        y_test.append(test_data[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test, scaler, training_data_len

def build_and_train_model(X_train, y_train, X_test, y_test, crypto):
    # Check if the model is already trained and saved
    model_path = f'models/{crypto}_model.h5'
    if os.path.exists(model_path):
        print(f"Loading existing model for {crypto}")
        model = load_model(model_path)
    else:
        # Initialize the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile and train the model
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, batch_size=1, epochs=10, validation_data=(X_test, y_test))

        # Save the trained model
        os.makedirs('models', exist_ok=True)
        model.save(model_path)

    return model


def update_and_train_model(model, new_data, crypto, epochs=10):
    # Preprocess the new data
    X_train, y_train, _, _, scaler, _ = preprocess_data(new_data)

    # Train the model with the new data
    tf.config.run_functions_eagerly(True)
    model.fit(X_train, y_train, batch_size=1, epochs=epochs)

    # Save the updated model
    model_path = f'models/{crypto}_model.h5'
    model.save(model_path)

    return model


def predict_and_evaluate(model, X_test, y_test, scaler, data, training_data_len, future_days):
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Calculate prediction accuracy
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    confidence = 0
    if 1 - mse / np.var(y_test) >= 0:
        confidence = np.sqrt(1 - mse / np.var(y_test))

    # Get the dates for the predicted data points
    X_test_dates = data.index[training_data_len + 1:].tolist()
    # Make predictions for future days
    future_predictions = []
    future_high_predictions = []
    future_low_predictions = []
    if future_days > 0:
        future_input = X_test[-1].reshape(1, -1, 1)

        for _ in range(future_days):
            future_prediction = model.predict(future_input)
            future_predictions.append(future_prediction[0])
            future_input = np.append(future_input[:, 1:], future_prediction).reshape(1, -1, 1)

            # Calculate possible high and low for future days based on historical data
            high_low_ratio = data['High'] / data['Low']
            avg_high_low_ratio = high_low_ratio.mean()
            future_high = future_prediction[0] * avg_high_low_ratio
            future_low = future_prediction[0] / avg_high_low_ratio
            future_high_predictions.append(future_high)
            future_low_predictions.append(future_low)

        future_predictions = scaler.inverse_transform(future_predictions)
        future_high_predictions = scaler.inverse_transform(future_high_predictions)
        future_low_predictions = scaler.inverse_transform(future_low_predictions)

    return predictions, mse, r2, confidence, X_test_dates, future_predictions, future_high_predictions, future_low_predictions


def best_buy_sell_time(future_predictions):
    future_predictions_list = [prediction[0] for prediction in future_predictions]

    buy_day, sell_day = 0, 0
    max_profit = 0

    for i in range(len(future_predictions_list)):
        for j in range(i + 1, len(future_predictions_list)):
            profit = future_predictions_list[j] - future_predictions_list[i]
            if profit > max_profit:
                max_profit = profit
                buy_day = i
                sell_day = j

    min_buy_price = future_predictions_list[buy_day]
    max_sell_price = future_predictions_list[sell_day]

    return buy_day, sell_day, min_buy_price, max_sell_price


def get_crypto_news_feed(crypto_list):
    news_feed = []

    for crypto in crypto_list:
        feed_url = f'https://news.google.com/rss/search?q={crypto}%20cryptocurrency'
        feed = feedparser.parse(feed_url)


        for entry in feed.entries[:10]:
            news_feed.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published
            })

    return news_feed


def display_predictions(predictions, chosen_crypto, predictions_text, future_days):
    predictions_text.delete('1.0', tk.END)
    pred_data = predictions[chosen_crypto]
    text = f"Predictions for {chosen_crypto}:\n\n"
    text += f"MSE: {pred_data['mse']:.5f}\n"
    text += f"R2: {pred_data['r2']:.5f}\n"
    text += f"Confidence: {pred_data['confidence']:.5f}\n\n"

    for i, pred in enumerate(pred_data['predictions']):
        text += f"Prediction {i + 1}: {pred[0]:.2f}\n"

    if future_days > 0:
        text += f"\nFuture predictions for the next {future_days} days:\n"
        for i, pred in enumerate(pred_data['future_predictions']):
            text += f"Prediction {i + 1}: {pred[0]:.2f}\n"

        # Add the best time to buy and sell along with profit and loss
        buy_day, sell_day, buy_price, sell_price = best_buy_sell_time(pred_data['future_predictions'])
        text += f"\nBest time to buy: Day {buy_day + 1} at a price of {buy_price:.2f}\n"
        text += f"Best time to sell: Day {sell_day + 1} at a price of {sell_price:.2f}\n"
        profit = sell_price - buy_price
        text += f"Anticipated profit: {profit:.2f}\n"

    predictions_text.insert(tk.END, text)



def display_news(crypto, news_text):
    news_text.delete('1.0', tk.END)
    news_feed = get_crypto_news_feed([crypto])

    for entry in news_feed:
        text = f"{entry['title']}\n{entry['published']}\n{entry['link']}\n\n"
        news_text.insert(tk.END, text)

def plot_candlestick_chart(crypto, start_date, end_date, predictions):
    data = get_crypto_data(crypto, start_date, end_date)

    # Add the moving average to the chart
    ma_period = 20  # Change this value to adjust the moving average period
    data['MA'] = data['Close'].rolling(window=ma_period).mean()

    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Add predicted data points to the chart
    predicted_data = go.Scatter(x=predictions[crypto]['X_test_dates'],
                                y=[pred[0] for pred in predictions[crypto]['predictions']],
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='red', size=5))
    fig.add_trace(predicted_data)

    # Add future predictions to the chart
    future_prediction_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(len(predictions[crypto]['future_predictions']))]
    future_predicted_data = go.Scatter(x=future_prediction_dates,
                                       y=[pred[0] for pred in predictions[crypto]['future_predictions']],
                                       mode='markers',
                                       name='Future Predictions',
                                       marker=dict(color='green', size=5))
    fig.add_trace(future_predicted_data)

    # Add the moving average to the chart
    ma_data = go.Scatter(x=data.index, y=data['MA'], name='Moving Average(20 Days)')
    fig.add_trace(ma_data)

    # Show the chart
    fig.show()



def create_ui_elements(root, predictions, news_feed, corr_matrix):
    def update_corr_labels(*args):
        sorted_corr = corr_matrix[chosen_crypto.get()].sort_values(ascending=False)
        positive_corr_list = sorted_corr[sorted_corr > 0][:10].index.tolist()
        positive_corr_label['text'] = f"Top 10 positively correlated cryptocurrencies with {chosen_crypto.get()}: {', '.join(positive_corr_list)}"

        sorted_corr = corr_matrix[chosen_crypto.get()].sort_values(ascending=True)
        negative_corr_list = sorted_corr[sorted_corr < 0][:10].index.tolist()
        negative_corr_label['text'] = f"Top 10 negatively correlated cryptocurrencies with {chosen_crypto.get()}: {', '.join(negative_corr_list)}"
    # Create frame for user input
    input_frame = ttk.Frame(root, padding="10")
    input_frame.grid(row=0, column=0, sticky=tk.W)

    # Create a dropdown to choose the cryptocurrency
    chosen_crypto = tk.StringVar()
    chosen_crypto.set(top_10_cryptos[0])
    crypto_dropdown = ttk.OptionMenu(input_frame, chosen_crypto, *top_10_cryptos)
    crypto_dropdown.grid(row=0, column=0, padx=(0, 10))
    chosen_crypto.trace('w', update_corr_labels)

    # Create a button to display predictions for the chosen cryptocurrency
    predictions_text = tk.Text(root, wrap=tk.WORD, width=50, height=20, padx=10, pady=10)
    predictions_text.grid(row=1, column=0, sticky=tk.W)

    # Add a new entry field to accept the number of future days
    future_days_label = ttk.Label(input_frame, text="Future days:")
    future_days_label.grid(row=1, column=0, padx=(0, 10), sticky=tk.E)

    future_days_entry = ttk.Entry(input_frame)
    future_days_entry.grid(row=1, column=1, padx=(0, 10), sticky=tk.W)

    show_predictions_button = ttk.Button(input_frame, text="Show Predictions",
                                         command=lambda: display_predictions(predictions, chosen_crypto.get(),
                                                                             predictions_text,
                                                                             int(future_days_entry.get())))
    show_predictions_button.grid(row=0, column=1, padx=(0, 10))


    # Create a button to display the news feed
    news_text = tk.Text(root, wrap=tk.WORD, width=50, height=20, padx=10, pady=10)
    news_text.grid(row=1, column=1, sticky=tk.W)

    show_news_button = ttk.Button(input_frame, text="Show News",
                                  command=lambda: display_news(chosen_crypto.get(), news_text))
    show_news_button.grid(row=0, column=2)
    # Create a button to display the candlestick chart
    show_chart_button = ttk.Button(input_frame, text="Show Chart",
                                   command=lambda: plot_candlestick_chart(chosen_crypto.get(), start_date, end_date, predictions))
    show_chart_button.grid(row=0, column=3)

    # Update the positively correlated cryptocurrencies
    sorted_corr = corr_matrix[chosen_crypto.get()].sort_values(ascending=False)
    positive_corr_list = sorted_corr[sorted_corr > 0][:10].index.tolist()
    positive_corr_label = ttk.Label(root,
                                    text=f"Top 10 positively correlated cryptocurrencies with {chosen_crypto.get()}: {', '.join(positive_corr_list)}")
    positive_corr_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

    # Update the negatively correlated cryptocurrencies
    sorted_corr = corr_matrix[chosen_crypto.get()].sort_values(ascending=True)
    negative_corr_list = sorted_corr[sorted_corr < 0][:10].index.tolist()
    negative_corr_label = ttk.Label(root,
                                    text=f"Top 10 negatively correlated cryptocurrencies with {chosen_crypto.get()}: {', '.join(negative_corr_list)}")
    negative_corr_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)


def main():
    # Define your top 10 cryptocurrencies and other application settings

    start_date = '2018-01-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    default_future_days=100

    # Fetch cryptocurrency data and preprocess it
    crypto_data = {}
    for crypto in top_10_cryptos:
        data = get_crypto_data(crypto, start_date, end_date)
        crypto_data[crypto] = preprocess_data(data)

    # Build and train the LSTM models for each cryptocurrency
    models = {}
    for crypto in top_10_cryptos:
        X_train, y_train, X_test, y_test, scaler, training_data_len = crypto_data[crypto]

        # Check if the model is already trained and saved
        model_path = f'models/{crypto}_model.h5'
        if os.path.exists(model_path):
            print(f"Loading existing model for {crypto}")
            model = load_model(model_path)
        else:
            model = build_and_train_model(X_train, y_train, X_test, y_test, crypto)

        models[crypto] = model
    # Update the models with the new data (if available)
    new_data_start_date = datetime.date.today()  # Change this date according to your needs
    new_data_end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    for crypto in top_10_cryptos:
        new_data = get_crypto_data(crypto, new_data_start_date, new_data_end_date)
        if not new_data.empty:
            print(f"Updating model for {crypto} with new data")
            model = models[crypto]
            model = update_and_train_model(model, new_data, crypto)
            models[crypto] = model
    # Predict the prices and calculate the prediction accuracy for each cryptocurrency
    predictions = {}
    for crypto in top_10_cryptos:
        model = models[crypto]
        X_test, y_test, scaler, data, training_data_len = crypto_data[crypto][2], crypto_data[crypto][3], \
            crypto_data[crypto][4], get_crypto_data(crypto, start_date, end_date), crypto_data[crypto][5]
        pred, mse, r2, confidence, X_test_dates, future_predictions, future_high_predictions, future_low_predictions = predict_and_evaluate(
            model, X_test, y_test,
            scaler, data,
            training_data_len,default_future_days)

        predictions[crypto] = {'predictions': pred, 'mse': mse, 'r2': r2, 'confidence': confidence,
                               'X_test_dates': X_test_dates, 'future_predictions': future_predictions,
                               'future_high_predictions': future_high_predictions,  # ADDED
                               'future_low_predictions': future_low_predictions}  # ADDED
    returns = {}
    for crypto in top_10_cryptos:
        data = get_crypto_data(crypto, start_date, end_date)
        returns[crypto] = data['Close'].pct_change().dropna()
    corr_matrix = pd.concat(returns, axis=1).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # Fetch the news RSS feed for the chosen top 10 cryptocurrencies
    news_feed = get_crypto_news_feed(top_10_cryptosrss)
    root = tk.Tk()
    root.title("Crypto Price Predictor")

    # Add the UI elements here
    create_ui_elements(root, predictions, news_feed, corr_matrix)

    root.mainloop()


if __name__ == "__main__":
    main()


