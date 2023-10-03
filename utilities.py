#!/usr/bin/env python
# coding: utf-8

# Build LSTM model for prediction

from contextlib import redirect_stdout
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time


import logging
# Configure the logging settings
logging.basicConfig(filename='error.log', level=logging.ERROR)

from sklearn.model_selection import train_test_split
import joblib


def assess_start(data, dataset):
    #find index in dataframe where data first reaches mean
    mean = data[dataset].mean()
    print(mean)
    
    for i in range(data.shape[0]):
        if data.iloc[i][dataset] > mean:
            return i
    return -1

def assess_end(data, dataset):
    #find last index where data is above mean
    mean = data[dataset].mean()
    print(mean)
    for i in range(data.shape[0]-1, -1, -1):
        if data.iloc[i][dataset] > mean:
            return i
    return -1

def prepare_data(data,dataset):
    #remove start and end of data
    start = assess_start(data, dataset)
    print('start', start)
    end = assess_end(data, dataset)
    print('end', end)
    return data[start:end]

def load_test_data(dataset):
    df = pd.read_csv('../data/'+ dataset +'day_15.csv')
    df = prepare_data(df, dataset)
    return df

def load_data_and_prepare(dataset, test=False):
    # Load data
    if not test:
        df = pd.read_csv('../data/unique_time/'+ dataset +'.csv')
    else:
        df = pd.read_csv('../data/'+ dataset +'/day_15.csv')
    
    # add columns to dataframe
    df.columns = ['time', dataset]
    df = prepare_data(df, dataset)

    # split data into train and test
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    return df_train, df_test


# Scale data
def scale_data(df, df_test, dataset, scaler=None):
    print('scaling')
    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(np.array(df[dataset]).reshape(-1, 1))
    test = scaler.transform(np.array(df_test[dataset]).reshape(-1, 1))
    return train, test, scaler

# Create dataset
def create_dataset(dataset, time_step=1, offset=0):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1-offset):
        # offset is the time period in between the last input sample point and output
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[(i + time_step + offset), 0])
    return np.array(dataX), np.array(dataY)

# Create the LSTM model
def create_model(time_step, model_specs=None):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    try:
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.4))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
    except ValueError as e:
            error_msg = f"Function: Creating Model, Error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
    return model

def train_model(model, X_train, y_train, X_val, y_val, dataset):
    import tensorflow as tf
    try:
        # save summary to file. create file if it doesn't exist
        if not os.path.isfile('./dir/models/'+ dataset +'/summary.txt'):
            open('./dir/models/'+ dataset +'/summary.txt', 'w+')
        with open('./dir/models/'+ dataset +'/summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        # set callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        # Train the model and save the history
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128, verbose=1, callbacks=[callback])
    except ValueError as e:
            error_msg = f"Function: Training model, Error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
    return history, model

def plot_loss(history):
    # Plot loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def predict(model, X_test, y_test, scaler):
    start_time = time.time()
    try:
        # Make predictions
        test_predict = model.predict(X_test)
        # Inverse transform to get the original scale
        if scaler is None:
            test_predict = test_predict.reshape(-1,1)
            y_test = y_test.reshape(-1,1)
        else:
            test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    except ValueError as e:
            error_msg = f"Function: Predicting, Error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Prediction execution time: {execution_time} seconds")
    print()
    return test_predict, y_test

def save_metrics(y_test, test_predict, dataset, file='performance_metrics.csv'):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    try:
        print(y_test.max(), test_predict.max())
        print('shapes: ', y_test.shape, test_predict.shape)

        # Calculate MAD performance metrics
        mad_test = np.mean(np.abs(y_test.reshape(-1, 1) - test_predict))
        print('Test MAD: %.3f' % mad_test)
        # Calculate MAE performance metrics
        mae_test = mean_absolute_error(y_test.reshape(-1, 1), test_predict)
        print('Test MAE: %.3f' % mae_test)
        # Calculate RMSE performance metrics
        rmse_test = np.sqrt(mean_squared_error(y_test.reshape(-1, 1), test_predict))
        print('Test RMSE: %.3f' % rmse_test)
        # Calculate MAPE performance metrics
        mape_test = np.mean(np.abs((y_test.reshape(-1, 1) - test_predict) / y_test.reshape(-1, 1))) * 100
        print('Test MAPE: %.3f' % mape_test)
        # Calculate R-squared performance metrics
        r2_test = 1 - (np.sum((y_test.reshape(-1, 1) - test_predict) ** 2) / np.sum((y_test.reshape(-1, 1) - np.mean(y_test.reshape(-1, 1))) ** 2))
        print('Test R2: %.3f' % r2_test)
        # Calculate adjusted R-squared performance metrics
        adj_r2_test = 1 - (1 - r2_test) * (len(y_test) - 1) / (len(y_test) - 1 - 1)
        print('Test adj R2: %.3f' % adj_r2_test)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # add entry to performance metrics file
        add_metrics = pd.DataFrame([{'dataset': dataset,'mad':str(mad_test), 'mae':str(mae_test), 'rmse': str(rmse_test), 'mape': str(mape_test), 'r2': str(r2_test), 'r2-adj': str(adj_r2_test), 'time_added': current_date}])
        add_metrics.to_csv(file, mode='a', index=False, header=False)
    except ValueError as e:
            error_msg = f"Function: Saving Metrics, Error: {str(e)}"
            print(error_msg)

def plot_prediction(y_test, test_predict, dataset):
    try:
    # Plot test prediction
        plt.figure(figsize=(15, 5))
        plt.plot(y_test.reshape(-1, 1), label='Actual')
        plt.plot(test_predict, label='Predicted')
        plt.title(dataset)
        plt.xlabel('Time')
        plt.ylabel(dataset)
        plt.legend()
        plt.show()
    except ValueError as e:
            error_msg = f"Function: Plotting Prediction, Error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)

def show_anomalies(y_test, test_predict, dataset):
    # show anomalies in test data where difference between actual and predicted is greater than the mean of the difference + 1 standard deviation
    diff = abs(test_predict - y_test)
    # calculate mean and standard deviation of y_test
    mean = np.mean(diff)
    std = np.std(y_test)
    diff_between_max_and_mean = y_test.max() - np.mean(y_test)
    print('mean: ', mean, 'std: ', std, 'diff: ', diff_between_max_and_mean)

    # break data into chunks of 60 * 20 minutes
    chunk_size = 60 * 20
    chunks = int(len(y_test) / chunk_size)
    print('chunks: ', chunks)

    # calculate mean of chunks
    chunk_means = []
    for i in range(chunks):
        chunk_means.append(np.mean(y_test[i*chunk_size:(i+1)*chunk_size]))

    # calculate mean of means
    mean_of_means = np.mean(chunk_means)
    print('mean of means: ', mean_of_means)

    # calculate standard deviation of means
    std_of_means = np.std(chunk_means)
    print('std of means: ', std_of_means)

    # calculate mean of means + 1 standard deviation of means
    mean_plus_std = mean_of_means + std_of_means
    print('mean + std: ', mean_plus_std)

    # find anomalies
    anomalies = []
    for i in range(len(test_predict)):
        if abs(test_predict[i] - y_test[i]) > (mean + (std*2) + 0.5):
            anomalies.append(i)


    # show anomalies on plot of actual and predicted values
    plt.figure(figsize=(15, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(test_predict, label='Predicted')
    plt.scatter(anomalies, (y_test)[anomalies], color='red', label='Anomaly')
    plt.title(dataset)
    plt.xlabel('Time')
    plt.ylabel(dataset)
    plt.legend()
    #savefig('anomalies.png')
    plt.savefig(dataset + 'anomalies.png')
    plt.show()

def get_data(dataset, time_step, offset=0, scaler=None, test=False, data=None, scale=True):
    try:
        if data is None:
            df, df_test = load_data_and_prepare(dataset, test)
        else:
            df, df_test = prepare_retraining_data(data, dataset)
        
        if scale:
            df, df_test, scaler = scale_data(df, df_test, dataset, scaler)
        else:
            df = np.array(df[dataset]).reshape(-1, 1)
            df_test = np.array(df_test[dataset]).reshape(-1, 1)
        X_train, y_train = create_dataset(df, time_step, offset)
        X_test, y_test = create_dataset(df_test, time_step, offset)

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    except ValueError as e:
            error_msg = f"Function: Collecting and Preparing Data, Error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)    
    return X_train, y_train, X_test, y_test, scaler

def test_metrics(dataset, time_step=20, offset=0, data=None):
    import tensorflow as tf
    #check if file exists
    if not os.path.isfile('./dir/scalers/'+ dataset+'_scaler.save'):
        scaler = None
    else:
        scaler = joblib.load('./dir/scalers/'+ dataset + '_scaler.save')
    
    start_time = time.time()  # Get the starting time

    X_train, y_train, X_test, y_test, scaler = get_data(dataset, time_step, offset, scaler, test=True, scale=scaler!=None, data=data)

    end_time = time.time()  # Get the ending time

    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Data processing execution time for testing metrics: {execution_time} seconds")
     
    #import model
    model = tf.keras.models.load_model('./dir/models/'+ dataset+'/1')
    print('model loaded')
    # Make predictions
    test_predict, y_test = predict(model, X_test, y_test, scaler)

    # Calculate RMSE performance metrics
    save_metrics(y_test, test_predict, dataset)

    return 'Model for ' + dataset  + ' tested'

def evaluate_model(data, dataset, time_step, offset):
    X_train, y_train, X_test, y_test, scaler = get_data(dataset, time_step, offset, scaler=None, test=False, data=data, scale=True)

    model = tf.keras.models.load_model('./dir/models/'+ dataset+'/1')
    print('model loaded')
    
    evaluation = model.evaluate(X_test, y_test)
    print(evaluation)
    # add to csv
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_metrics = pd.DataFrame([{'dataset': dataset,'loss':str(evaluation[0]), 'accuracy':str(evaluation[1]), 'time_added': current_date}])
    add_metrics.to_csv('./dir/models/evaluation_metrics.csv', mode='a', index=False, header=False)


def build_model(dataset, time_step=20, offset=0, data=None, model_specs=None):
    start_time = time.time()

    X_train, y_train, X_test, y_test, scaler = get_data(dataset, time_step, offset, scaler=None, test=False, data=data, scale=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    end_time = time.time()  # Get the ending time

    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Data processing execution time for building model: {execution_time} seconds")
     
    # Create the LSTM model
    model = create_model(time_step, model_specs)
    
    # Train the model and save the history
    history, model = train_model(model, X_train, y_train, X_val, y_val, dataset)

    #save model
    model.save('./dir/models_new/'+ dataset)
    
    timenow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #save scaler
    joblib.dump(scaler, './dir/models_new/'+ dataset + '/scaler_'+timenow+'.save')

    # Make predictions
    test_predict, y_test = predict(model, X_test, y_test, scaler)

    # Calculate RMSE performance metrics
    # save_metrics(y_test, test_predict, dataset, './dir/models_new/performance_metrics.csv')

    # Plot test prediction
    # plot_prediction(y_test, test_predict, dataset)

    # show anomalies in test data
    # show_anomalies(y_test, test_predict, dataset)

    return 'Model for ' + dataset  + ' built and saved'


def retrain_ml_model(data, dataset, time_step, offset):
    model = tf.keras.models.load_model('./dir/models/'+ dataset+'/1')
    
    if not os.path.isfile('./dir/scalers/'+ dataset+'_scaler.save'):
        scaler = None
    else:
        scaler = joblib.load('./dir/scalers/'+ dataset + '_scaler.save')

    start_time = time.time()  # Get the starting time

    X_train, y_train, X_test, y_test, scaler = get_data(dataset, time_step, offset, scaler, test=True, data=data, scale=scaler!=None)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    end_time = time.time()  # Get the ending time

    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Data processing execution time for retraining model: {execution_time} seconds")

    # Model performance before retraining
    test_predict, y_test = predict(model, X_test, y_test, scaler)
    save_metrics(y_test, test_predict, dataset+'-current', './dir/models_retrained/comparison_metrics.csv')
    
    # Train the model and save the history
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1, callbacks=[callback])

    #save model
    model.save('./dir/models_retrained/'+ dataset)

    test_predict, y_test = predict(model, X_test, y_test, scaler)
    save_metrics(y_test, test_predict, dataset+'-retrained', './dir/models_retrained/comparison_metrics.csv')

    return 'Model for ' + dataset  + ' retrained and saved'


def prepare_retraining_data(data, dataset):
    try:
        if data.shape[1] == 2:
            data.columns = ['time', dataset]
        else:
            data.columns = [dataset, 'error', 'status', 'nano', 'time']
            data.drop(['error', 'status', 'nano'], axis=1, inplace=True)
        print('here', data.head())
        data['time'] = pd.to_datetime(data['time'])
        data = prepare_data(data, dataset)

        train, test = train_test_split(data, test_size=0.2, shuffle=False)
    except ValueError as e:
            error_msg = f"Function: Preparing Training Data, Error: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
    return train, test