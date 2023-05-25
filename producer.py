import json
import sys
from kafka import KafkaConsumer, KafkaProducer
import csv
from time import sleep
import datetime

bootstrap_servers = ['localhost:9092']
topicName = 'velocity'
group_id = None
producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10), value_serializer=lambda v: json.dumps(v).encode('utf-8'))


with open('day_6.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader) # Get the header row

    for row in reader:
        data = dict(zip(headers, row)) # Convert each row to a dictionary
        producer.send(topicName, value=data) # Replace 'test-topic' with your desired topic name
        print(data)
        sleep(1)

# import csv
# import quixstreams as qx
# import time
# import datetime
# import math


# # Connect to your kafka client
# client = qx.KafkaStreamingClient('127.0.0.1:9092')

# # Open the output topic which is where data will be streamed out to
# topic_producer = client.get_topic_producer("mytesttopic")

# # Set stream ID or leave parameters empty to get stream ID generated.
# stream_vel = topic_producer.create_stream(stream_id='velocity')
# stream_tor = topic_producer.create_stream(stream_id='torque')
# # stream_vel.properties.name = "Hello World Python stream"

# with open('day_6.csv', 'r') as file:
#     reader = csv.reader(file)
#     headers = next(reader) # Get the header row

#     for row in reader:
#         stream_vel.timeseries \
#             .buffer \
#             .add_timestamp(datetime.datetime.utcnow()) \
#             .add_value(value=str(row), parameter_id="velocity") \
#             .publish()
#         print(row)
#         time.sleep(1)

# print("Closing stream")
# stream_vel.close()