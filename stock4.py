import datetime
import yfinance as yf
from finta import TA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt


stockTicker = "ACCON.ST"
stockTicker = "PLUG"
stockTicker = "ETH-USD" # 350 -1h - 12 - 200 epochs
stockTicker = "BTC-USD" # 350 -1h - 12 - 200 epochs
stockTicker = "ERIC-B.ST" # 350 - 1d - 1 - 200 epochs
stockTicker = "KINV-B.ST"
stockTicker = "INVE-B.ST"
stockTicker = "ALIV-SDB.ST"
stockTicker = "SKF-B.ST"
stockTicker = "TEL2-B.ST"
stockTicker = "SWED-A.ST"


NUM_DAYS = 50     # The number of days of historical data to retrieve
INTERVAL = '15m'     # Sample rate of historical data
INDICATORS = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

window = 12
nr_cells_lstm = 256
noOfEpochs = 200
batchsize = 40
testSize = 0.3
confidenseThreshold = 0.8

def get_stock_data(tickerSymbol):
    # List of symbols for technical indicators

    """
    Next we pull the historical data using yfinance
    Rename the column names because finta uses the lowercase names
    """

    start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )
    end = datetime.datetime.today()
    data = yf.download(tickerSymbol, start=start, end=end, interval=INTERVAL)
    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
    return data


def _get_indicator_data(data):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """

    for indicator in INDICATORS:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(15).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['Adj Close'])
    
    return data

def get_ema_history_data(data,emaData,dataColumns,columnIndex):
    len = 10
    counter = 0
    dataNewEma = pd.DataFrame()
    for ind in emaData.index[:-9]:
        #Get last x of ticks. Must include history in training
        data2 = data.iloc[counter:len,columnIndex]
        #Extract the date index
        index = data2.index
        #Convert to a list
        a_list = list(index)
        # Take the last day of the index
        a_list = a_list[9:10]
        # Set the index on the date that should include history data
        data2List = data2.tolist() 
        # Transpose the EMA history. (Rows -> Colums)
        data2 = pd.DataFrame(data2List).T
        try:
            data2.columns = dataColumns
            data2.index = a_list
        except:
            print(data2)
            print(dataNewEma)

        dataNewEma = pd.concat([dataNewEma,data2])
        len +=1
        counter +=1
    return(dataNewEma)


def addHistoryDataForEachSet(data):
    historyData = data.loc[:,"close"]
    dataColumns = ['close-1','close-2','close-3','close-4','close-5','close-6','close-7','close-8','close-9','close-1+']
    dataNew = get_ema_history_data(data,historyData,dataColumns,0)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 

    historyData = data.loc[:,"ema5"]
    dataColumns = ['ema5-1','ema5-2','ema5-3','ema5-4','ema5-5','ema5-6','ema5-7','ema5-8','ema5-9','ema5-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,7)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data)
    
    historyData = data.loc[:,"ema15"]
    dataColumns = ['ema15-1','ema15-2','ema15-3','ema15-4','ema15-5','ema15-6','ema15-7','ema15-8','ema15-9','ema15-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,6)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data)

    historyData = data.loc[:,"ema21"]
    dataColumns = ['ema21-1','ema21-2','ema21-3','ema21-4','ema21-5','ema21-6','ema21-7','ema21-8','ema21-9','ema21-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,5)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data)

    historyData = data.loc[:,"ema50"]
    dataColumns = ['ema50-1','ema50-2','ema50-3','ema50-4','ema50-5','ema50-6','ema50-7','ema50-8','ema50-9','ema50-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,4)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data)

    historyData = data.loc[:,"14 period ATR"]
    dataColumns = ['ATR-1','ATR-2','ATR-3','ATR-4','ATR-5','ATR-6','ATR-7','ATR-8','ATR-9','ATR-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,2)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 

    historyData = data.loc[:,"14 period RSI"]
    dataColumns = ['RSI-1','RSI-2','RSI-3','RSI-4','RSI-5','RSI-6','RSI-7','RSI-8','RSI-9','RSI-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,1)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 

    historyData = data.loc[:,"MACD"]
    dataColumns = ['MACD-1','MACD-2','MACD-3','MACD-4','MACD-5','MACD-6','MACD-7','MACD-8','MACD-9','MACD-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,3)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 

    historyData = data.loc[:,"normVol"]
    dataColumns = ['VOL-1','VOL-2','VOL-3','VOL-4','VOL-5','VOL-6','VOL-7','VOL-8','VOL-9','VOL-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,7)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 

    return data

def _produce_prediction(data, window):
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    """
    
    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]

    # Add new column for prediction
    data['pred'] = prediction.astype(int)
    
    return data


# Download stock data
df = get_stock_data(stockTicker)
df = _get_indicator_data(df)
df = addHistoryDataForEachSet(df)
num_columns = df.shape[1] # Needs to be specified in the model


# Add a column for the label
df['label'] = df['close'].shift(-window) > df['close']
df['result'] = (df['close'].shift(-window) - df['close'])/df['close'] * 100

print(df)
# Drop rows with missing values
df.dropna(inplace=True)
#df = df.drop(["Stock Splits","Dividends"], axis='columns')

# Scale the data
print(df.shape)
scaler = MinMaxScaler()
df_X = df.drop(["label","result"], axis='columns')
df_X = (df_X-df_X.min())/(df_X.max()-df_X.min())
df_X = pd.DataFrame(scaler.fit_transform(df_X),columns=df_X.columns)
# Reshape the input data for the LSTM
#df_X = df_X[['Open','Close', 'Volume', 'High', 'Low']].values
df_X = df_X.values
X = np.reshape(df_X, (df_X.shape[0], 1, num_columns))
#X = np.reshape(df_X[['Close', 'Volume', 'High', 'Low']], (df_X.shape[0], 1, 4))
df_reset = df.reset_index()
print(df_reset)
y = df_reset[['label','index','result']]
# Split the data into training and test sets
#X = df[['Close', 'Volume', 'High', 'Low']]
#y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)

# Reshape the input data for the LSTM
y_train = y_train.drop(["index","result"], axis='columns')
print(y_test)
y_test_result = y_test
y_test = y_test.drop(["index","result"], axis='columns')
print(y_test)

#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Initialize the LSTM model
model = Sequential()

# In summary, this line of code is adding a layer with 128 
# LSTM cells to the model, # and it is specifying that the 
# input data to this layer will have the shape of (number of samples, 1, 5).
model.add(LSTM(nr_cells_lstm, return_sequences=True, input_shape=(1, num_columns)))

# adding more memory to the LSTM model is to use stacked LSTM layers, 
# it means to use multiple layers with the return_sequences set to True, 
# this allows the information to be passed from one layer to the other and 
# increase the capacity of the model to store information.
model.add(LSTM(nr_cells_lstm,return_sequences=True,))
model.add(LSTM(nr_cells_lstm,return_sequences=True,))
model.add(LSTM(nr_cells_lstm))

# Use sigmoid when binary results
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=noOfEpochs, batch_size=batchsize, verbose=1)


# Get the model's predictions on the test data
# Run the model on the test data
predictions = model.predict(X_test)
predictions_class = (predictions > 0.5).astype(int)

nrCorrect = 0
total = 0
print(y_test)
# Interpret the results
for i in range(len(predictions)):
    if predictions[i] > confidenseThreshold:
        if predictions_class[i] == 1 and y_test.iloc[i]['label'] == True or predictions_class[i] == 0 and y_test.iloc[i]['label'] == False:
            nrCorrect += 1
        total += 1
        print("Prediction:", predictions_class[i], "Confidence level:", predictions[i], "Correct Result:", round(y_test_result.iloc[i]['result'],4),"%","Date",y_test_result.iloc[i]['index'])
        print("Total: ",total," Correct: ",nrCorrect)

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