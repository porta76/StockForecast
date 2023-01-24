import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from finta import TA
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, accuracy_score

# Use this to normalize data that is necessary when training on stocks
from sklearn.preprocessing import StandardScaler
"""
Defining some constants for data mining
"""

NUM_DAYS = 5000     # The number of days of historical data to retrieve
INTERVAL = '1d'     # Sample rate of historical data
#symbol = 'ABB'      # Symbol of the desired stock
symbol = '^GSPC'      # Symbol of the desired stock
symbol2 = '^OMX'
#symbol = 'INVE-B.ST'
#symbol2 = 'ERIC-B.ST'
#symbol ='MINEST.ST'

# List of symbols for technical indicators
#INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']
INDICATORS = ['RSI', 'ATR','MACD']
"""
Next we pull the historical data using yfinance
Rename the column names because finta uses the lowercase names
"""

start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )
end = datetime.datetime.today()

data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
data2 = yf.download(symbol2, start=start, end=end, interval=INTERVAL)
data2.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
print(len(data))

# tmp = data.iloc[-60:]
# tmp['close'].plot()
# plt.show()

"""
Next we clean our data and perform feature engineering to create new technical indicator features that our
model can learn from
"""

def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """
    
    return data.ewm(alpha=alpha).mean()

#data = _exponential_smooth(data, 0.65)

# tmp1 = data.iloc[-60:]
# tmp1['close'].plot()
# plt.show()

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

data = _get_indicator_data(data)
data2 = _get_indicator_data(data2)
print(data.columns)

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


    historyData = data.loc[:,"close"]
    dataColumns = ['close-1','close-2','close-3','close-4','close-5','close-6','close-7','close-8','close-9','close-1+']
    dataNew = get_ema_history_data(data,historyData,dataColumns,0)
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
    """
    historyData = data.loc[:,"VIp"]
    dataColumns = ['VIp-1','VIp-2','VIp-3','VIp-4','VIp-5','VIp-6','VIp-7','VIp-8','VIp-9','VIp-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,14)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 
    """
    
    historyData = data.loc[:,"normVol"]
    dataColumns = ['VOL-1','VOL-2','VOL-3','VOL-4','VOL-5','VOL-6','VOL-7','VOL-8','VOL-9','VOL-10']
    dataNew = get_ema_history_data(data,historyData,dataColumns,7)
    data = data.merge(dataNew, left_index=True, right_index=True)
    print(data) 

    return data

data = addHistoryDataForEachSet(data)
data2 = addHistoryDataForEachSet(data2)

# Select data. Using index location last 16 to last 11. Hmm Why exactly those??
live_pred_data = data.iloc[-26:-11]

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

data = _produce_prediction(data, window=2)
data2 = _produce_prediction(data2, window=2)
#del (data['close'])
data = data.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here
data2 = data2.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here
print(len(data))

# Remove common movements keep more extreme
#data = data[data['14 period ATR'] / data['close'] > 0.02]
#data = data[data['14 period RSI'] < 40]

print(data.columns)
print(len(data))

def _train_random_forest(X_train, y_train, X_test, y_test):

    """
    Function that uses random forest classifier to train the model
    :return:
    """
    
    # Create a new random forest classifier
    # rf = RandomForestClassifier()
    rf = RandomForestClassifier(max_features="sqrt")
    
    # Dictionary of all values we want to test for n_estimators

    #params_rf = {'n_estimators': [110,130,140,150,160,180,200,300]}
    #params_rf = {'n_estimators': [300,520,740,960,1200]}
    #params_rf = {'n_estimators': [300,400,500,600,700]}
    #params_rf = {'n_estimators': [100,120,140,160,170]}
    params_rf = {'n_estimators': [520]}
    # Use gridsearch to test all values for n_estimators
    #rf_gs = GridSearchCV(rf, params_rf, cv=5)
    rf_gs = GridSearchCV(rf, params_rf, cv=10)

    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    # Save best model
    rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    print(rf_gs.best_params_)

    # Try to get rid of infinite numbers

    prediction = rf_best.predict(X_test)

    #print(classification_report(y_test, prediction))
    print(classification_report(y_test, prediction,zero_division=1))
    print(confusion_matrix(y_test, prediction))    
    return rf_best
    
##rf_model = _train_random_forest(X_train, y_train, X_test, y_test)

def _train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 75)} # default 25
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)

    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    # Save best model
    knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    print(knn_gs.best_params_)
    
    prediction = knn_best.predict(X_test)

    #print(classification_report(y_test, prediction))
    print(classification_report(y_test, prediction,zero_division=1))
    print(confusion_matrix(y_test, prediction))
    
    return knn_best
    
    
##knn_model = _train_KNN(X_train, y_train, X_test, y_test)
def _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test):
#def _ensemble_model(rf_model, X_train, y_train, X_test, y_test):
#def _ensemble_model(knn_model, X_train, y_train, X_test, y_test):
    
    # Create a dictionary of our models
    # estimators=[('knn', knn_model), ('rf', rf_model), ('gbt', gbt_model)]
    estimators=[('knn', knn_model), ('rf', rf_model)]
    #estimators=[('rf', rf_model)]
    #estimators=[('knn', knn_model)]
    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')
    
    #fit model to training data
    ensemble.fit(X_train, y_train)
    
    #test our model on the test data
    print(ensemble.score(X_test, y_test))
    
    prediction = ensemble.predict(X_test)

    #print(classification_report(y_test, prediction))
    print(classification_report(y_test, prediction,zero_division=1))
    print(confusion_matrix(y_test, prediction))
    
    return ensemble
    
##ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test)

def cross_Validation(data):

    # Split data into equal partitions of size len_train
    
    num_train = 100 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 400 # Length of each train-test set
    
    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    ensemble_RESULTS = []
    
    # Prepare to plot learning curve
    rf_train_sizes = []
    rf_train_scores = []
    rf_test_scores = []

    i = 0
    while True:
        
        # Partition the data into chunks of size len_train every num_train days
        df = data.iloc[i * num_train : (i * num_train) + len_train]
        i += 1
        print(i * num_train, (i * num_train) + len_train)
        
        if len(df) < len_train:
            break

        y = df['pred']
        features = [x for x in df.columns if x not in ['pred']]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 7 * len(X) // 10,shuffle=False)
        # Test this to get rid of infinity or large number in df
        # printing column name where infinity is present
        print()
        print("printing column name where infinity is present")
        col_name = X_test.columns.to_series()[np.isinf(df).any()]
        print(col_name)        
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test = X_test.fillna(X_test.mean())
        
        # Normalize
        #print('Before normalization: ',X_train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #print('Normalized data: ',X_train)


        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        knn_model = _train_KNN(X_train, y_train, X_test, y_test)
        ensemble_model = _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test)
        #ensemble_model = _ensemble_model(rf_model, X_train, y_train, X_test, y_test)
        #ensemble_model = _ensemble_model(knn_model, X_train, y_train, X_test, y_test)
        rf_prediction = rf_model.predict(X_test)
        knn_prediction = knn_model.predict(X_test)
        ensemble_prediction = ensemble_model.predict(X_test)
        print('rf prediction is ', rf_prediction)
        print('knn prediction is ', knn_prediction)
        print('ensemble prediction is ', ensemble_prediction)
        print('truth values are ', y_test.values)
        
        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)
        ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)
        
        print(rf_accuracy, knn_accuracy, ensemble_accuracy)
        #print(knn_accuracy, ensemble_accuracy)
        rf_RESULTS.append(rf_accuracy)
        knn_RESULTS.append(knn_accuracy)
        ensemble_RESULTS.append(ensemble_accuracy)

        # Collect the results from each iteration
        rf_train_sizes.append(len(X_train))
        rf_train_scores.append(rf_model.score(X_train, y_train))
        rf_test_scores.append(rf_accuracy)


        
    # Plot the learning curve
    plt.figure()
    plt.title("Random Forest Learning Curve")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(rf_train_sizes, rf_train_scores, 'o-', color="r", label="Training Accuracy")
    plt.plot(rf_train_sizes, rf_test_scores, 'o-', color="g", label="Cross-Validation Accuracy")
    plt.legend(loc="best")
    plt.show()
        
    #print('RF Accuracy = ' + str( sum(rf_RESULTS) / len(rf_RESULTS)))
    #print('KNN Accuracy = ' + str( sum(knn_RESULTS) / len(knn_RESULTS)))
    print('Ensemble Accuracy = ' + str( sum(ensemble_RESULTS) / len(ensemble_RESULTS)))
    #del(live_pred_data['close'])
    
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit the scaler to the training data
    scaler.fit(live_pred_data)
    # Transform the live data using the fitted scaler
    live_pred_data_scaled = scaler.transform(live_pred_data)
    prediction = ensemble_model.predict(live_pred_data_scaled)
    print(live_pred_data.head())
    print(prediction)
    
    df_accuracy = pd.DataFrame(rf_RESULTS, columns=['Accuracy'])
    df_accuracy['Accuracy'].plot()
    plt.show()
    print(knn_RESULTS)

cross_Validation(data)
cross_Validation(data2)
