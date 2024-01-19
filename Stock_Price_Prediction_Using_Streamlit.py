import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model('C:/Users/hp/Python_Projects/Resume_Projects/Stock Price Prediction Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2013-01-01'
end = '2013-12-31'

stock_data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(stock_data)

train_data = pd.DataFrame(stock_data.Close[:int(len(stock_data)*0.8)])
test_data = pd.DataFrame(stock_data.Close[int(len(stock_data)*0.8):])

# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
mmc = MinMaxScaler(feature_range=(0, 1))

pas_100_days = train_data.tail(100)
test_data = pd.concat([pas_100_days, test_data], ignore_index=True)
scale_test_data = mmc.fit_transform(test_data)

# Price vs Moving Around 50 days Diagram
st.subheader('Price vs MA50')
ma_50_days = stock_data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(stock_data.Close, 'g')
plt.show()
st.pyplot(fig1)

# Price vs Moving Around 50 days vs Moving Around 100 days Diagram
st.subheader('Price vs MA50 vs MA100')
ma_100_days = stock_data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(stock_data.Close, 'g')
plt.show()
st.pyplot(fig2)

# Price vs Moving Around 100 days vs Moving Around 200 days Diagram
st.subheader('Price vs MA100 vs MA200')
ma_200_days = stock_data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(stock_data.Close, 'g')
plt.show()
st.pyplot(fig3)

X, Y = [], []
for i in range(100, scale_test_data.shape[0]):
    X.append(scale_test_data[i-100:i])
    Y.append(scale_test_data[i, 0])

X, Y = np.array(X), np.array(Y)

# prediction of given data
Y_pred = model.predict(X)
# find scale from min-max scaling
scale = 1/mmc.scale_
Y_pred *= scale
Y *= scale

# Original Price vs Predicted Price Diagram
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(Y_pred, 'r', label='Predicted Price')
plt.plot(Y, 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
