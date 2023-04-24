from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, ExcelFile, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Load data
f = r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\data\a Processed data 2023_2.xlsx"
xl = ExcelFile(f)



list_region = []
list_pred = []
list_test = []
for name in xl.sheet_names:
# for name in ['Bundy', 'Lismore', 'SEQ']:
    dataset = xl.parse(name)
    # df_23 = dataset.loc[dataset['Year'] == 2023]
    df_22 = dataset.loc[dataset['Year'] == 2022]

    # print(dataset.columns)
    dataset.drop(['Year', ' Tonnes'], axis=1, inplace=True)
    first_column = dataset.pop(' Dev%')
    dataset.insert(0, 'Dev%', first_column)

    # df_23_X = df_23.drop(['Year', ' Tonnes', ' Dev%'], axis=1)
    # X_23 = df_23_X.values
    y_22 = df_22[' Dev%'].values
    df_22_X = df_22.drop(['Year', ' Tonnes', ' Dev%'], axis=1)
    X_22 = df_22_X.values
    

    dataset.drop(dataset.tail(2).index,inplace=True)
    # print(dataset) 
    values = dataset.values

    # ensure all data is float
    values = values.astype('float32')
    X = values[:, 1:]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    y = values[:, 0]

    # # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(values) 

    # split into train and test sets
    # values = scaled
    n_train_years = 16
    train = values[:n_train_years, :]
    test = values[n_train_years:, :]

    # split into input and output
    train_X, train_y = train[:, 1:], train[:, 0]
    test_X, test_y = test[:, 1:], test[:, 0]



    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) # (16, 1, 67) (16,) (7, 1, 67) (7,)
 

    # X_23 = X_23.reshape((X_23.shape[0], 1, X_23.shape[1]))
    X_22 = X_22.reshape((X_22.shape[0], 1, X_22.shape[1]))




    # models

    m_rnn = Sequential()
    m_rnn.add(LSTM(40, input_shape=(train_X.shape[1],  train_X.shape[2])))
    m_rnn.add(Dense(10))
    m_rnn.add(Dense(1))
    m_rnn.compile(loss='mean_squared_error', optimizer='adam')
    m_rnn.fit(train_X, train_y, epochs=100, batch_size=5, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # history = rnn.fit(train_X, train_y, epochs=50, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # plot history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()


    # make a prediction


    yhat_rnn = m_rnn.predict(test_X)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(test_y, yhat_rnn))
    print('LSTM Test RMSE: %.3f' % rmse)


    # # predict 2023
    # yhat_2023 = model.predict(X_23)
    # print('2023 Dev', yhat_2023[0]) #2023 Dev [[-1.1814936]]

    # list_region.append(name)
    # list_pred.append(yhat_2023[0][0])

    # predict 2022
    yhat_2022 = m_rnn.predict(X_22)
    print('2022 Dev', yhat_2022[0]) 

    list_region.append(name)
    list_pred.append(yhat_2022[0][0])
    list_test.append(y_22[0])

data = {'Region': list_region,
        'Pred(LSTM)': list_pred,
        'y':list_test}
df = DataFrame(data)
df.to_csv(r'C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\output\predictions_2022_lstm.csv', index=False)