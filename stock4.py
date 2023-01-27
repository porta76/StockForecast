import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

nr_cells_lstm = 64
noOfEpochs = 50
batchsize = 32

# Download stock data
stock = yf.Ticker("ERIC-B.ST")
stock = yf.Ticker("ACCON.ST")
stock = yf.Ticker("MINEST.ST")
stock = yf.Ticker("PLUG")

df = stock.history(period="max",interval="1d")
print(df.shape)
# Add a column for the label
df['label'] = df['Close'].shift(-1) > df['Close']
print(df.shape)
# Drop rows with missing values
df.dropna(inplace=True)
print(df)
df = df.drop(["Stock Splits","Dividends"], axis='columns')

# Scale the data
print(df.shape)
scaler = MinMaxScaler()
df_X = df.drop(["label"], axis='columns')
df_X[['Open','Close', 'Volume', 'High', 'Low']] = scaler.fit_transform(df_X[['Open','Close', 'Volume', 'High', 'Low']])

# Reshape the input data for the LSTM
print(df_X)
print(df_X.shape)
df_X = df_X[['Open','Close', 'Volume', 'High', 'Low']].values
X = np.reshape(df_X, (df_X.shape[0], 1, 5))
#X = np.reshape(df_X[['Close', 'Volume', 'High', 'Low']], (df_X.shape[0], 1, 4))
y = df['label']

# Split the data into training and test sets
#X = df[['Close', 'Volume', 'High', 'Low']]
#y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape the input data for the LSTM
print(X_train.shape)
#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Initialize the LSTM model
model = Sequential()

# In summary, this line of code is adding a layer with 128 
# LSTM cells to the model, # and it is specifying that the 
# input data to this layer will have the shape of (number of samples, 1, 5).
model.add(LSTM(nr_cells_lstm, return_sequences=True, input_shape=(1, 5)))

# adding more memory to the LSTM model is to use stacked LSTM layers, 
# it means to use multiple layers with the return_sequences set to True, 
# this allows the information to be passed from one layer to the other and 
# increase the capacity of the model to store information.
model.add(LSTM(nr_cells_lstm))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=noOfEpochs, batch_size=batchsize, verbose=1)

# Evaluate the model on the test data
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()