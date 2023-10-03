import datetime
import json
import os
import threading
from kafka import KafkaProducer, KafkaConsumer
import time
import pandas as pd
from predictions import prediction

# if os.path.exists('./data/velocity_calc.csv'):
#     velocity_calc = pd.read_csv('./data/velocity_calc.csv')
# else:
#     print('velocity file not found')

if os.path.exists('./data/torque.csv'):
    torque = pd.read_csv('./data/torque.csv')
else:
    print('torque file not found')

torque = torque.drop(['nano', 'error', 'status'], axis=1)
# velocity = velocity.drop(['nano', 'error', 'status'], axis=1)
# velocity_calc = velocity_calc.drop(['nano', 'error', 'status'], axis=1)

# set up Kafka producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10),value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# function to send data
def send_data():
    n = 0
    while True:
        # timestamp = datetime.datetime.utcnow()
        torque_data = dict(zip(['torque', 'time'], torque.iloc[n]))
        # velocity_data = dict(zip(['velocity', 'time'], velocity.iloc[n]))
        # velocity_calc_data = dict(zip(['velocity_calc', 'time'], velocity_calc.iloc[n]))
        print('adding data to kafka')
        producer.send('torque', torque_data)
        # producer.send('velocity', velocity_data)
        # producer.send('velocity_calc', velocity_calc_data)
        n+=1
        time.sleep(0.1)

# function to consume data and send it to prediction function
def consume_data():
    consumer = KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='latest', api_version=(0, 10), consumer_timeout_ms=10000, fetch_max_wait_ms=1000, value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    consumer.subscribe(['torque', 'velocity', 'velocity_calc'])
    print('consumer subscribed')
    # if not os.path.exists('./data/velocity_calc_streamed.csv'):
    #     velocity_calc_data = pd.DataFrame()
    #     velocity_calc_data.to_csv('velocity_calc_streamed.csv', index=False)

    if not os.path.exists('./data/torque_streamed.csv'):
        torque_data = pd.DataFrame()
        torque_data.to_csv('torque_streamed.csv', index=False)

    while True:
        for message in consumer:
            # if message.topic == 'velocity_calc':
            #     # create dict from message value
            #     new_row = pd.DataFrame([message.value])
            #     new_row.to_csv('velocity_calc_streamed.csv', mode='a', index=False, header=False)
            if message.topic == 'torque':
                new_row = pd.DataFrame([message.value])
                new_row.to_csv('torque_streamed.csv', mode='a', index=False, header=False)

def predict():
    # check if file exists
    while True:
        velocity_calc_data = pd.DataFrame()
        try:
            velocity_calc_data = pd.read_csv('./velocity_calc_streamed.csv')
            time.sleep(1)
        except:
            print('velcalc file not found')
            time.sleep(1)
        
        torque_data = pd.DataFrame()
        try:
            torque_data = pd.read_csv('./velocity_calc_streamed.csv')
            time.sleep(1)
        except:
            print('velcalc file not found')
            time.sleep(1)

        if len(torque_data) > 30:
            n = 0
            # if not os.path.exists('predictions_velocity_calc.csv'):
            #     predictions = pd.DataFrame()
            #     predictions.to_csv('predictions_velocity_calc.csv', index=False)
            if not os.path.exists('predictions_torque'):
                predictions = pd.DataFrame()
                predictions.to_csv('predictions_torque.csv', index=False)
            while True:
                print('predicting')
                # read the last 30 rows from csv file
                torque_data = pd.read_csv('torque_streamed.csv', nrows=30, skiprows=range(1, n))
                # add column names
                torque_data.columns = ['torque', 'time']

                # velocity_calc_data = pd.read_csv('velocity_calc_streamed.csv', nrows=20, skiprows=range(1, n))
                # velocity_calc_data.columns = ['velocity_calc', 'time']

                # prediction(velocity_calc_data, 'velocity_calc', 'velocity_calc')
                prediction(torque_data, 'torque', 'torque')
                n+=1
                time.sleep(0.1)

def exit_condition():
    # set up exit condition
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            producer_thread.join()
            consumer_thread.join()
            exit(0)

# create empty csv
# torque_data = pd.DataFrame()
# torque_data.to_csv('torque_streamed.csv', index=False)
# velocity_data = pd.DataFrame()
# velocity_data.to_csv('velocity_streamed.csv', index=False)
# velocity_calc_data = pd.DataFrame()
# velocity_calc_data.to_csv('velocity_calc_streamed.csv', index=False)


# start producer and consumer threads
producer_thread = threading.Thread(target=send_data)
consumer_thread = threading.Thread(target=consume_data)

predict_thread = threading.Thread(target=predict)
exit_thread = threading.Thread(target=exit_condition)

producer_thread.start()
consumer_thread.start()
predict_thread.start()
exit_thread.start()
