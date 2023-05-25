import json
from kafka import KafkaConsumer
import sys
import datetime

bootstrap_servers = ['localhost:9092']
topicName = 'velocity'
group_id = None

consumer = KafkaConsumer(
    topicName,
    bootstrap_servers=bootstrap_servers,
    group_id=group_id,
    api_version=(0, 10)
)

consumer.subscribe(['torque', 'velocity', 'motor_temperature'])
print("Consumer is ready to receive messages")

try:
    print(datetime.datetime.utcnow())
    for msg in consumer:
        print(msg.value)
except KeyboardInterrupt:
    consumer.close()
    sys.exit()

# import sys
# import quixstreams as qx
# import pandas as pd
# from cancel_token import CancellationToken

# # Connect to your kafka client
# client = qx.KafkaStreamingClient('127.0.0.1:9092')

# # get the topic consumer for a specific consumer group
# topic_consumer = client.get_topic_consumer(topic = "mytesttopic",
#                                            consumer_group = "empty-destination")


# def on_dataframe_received_handler(stream_consumer: qx.StreamConsumer, df: pd.DataFrame):
#     # do something with the data here
#     print(df['velocity'].values[0])


# def on_stream_received_handler(stream_consumer: qx.StreamConsumer):
#     # subscribe to new DataFrames being received
#     # if you aren't familiar with DataFrames there are other callbacks available
#     # refer to the docs here: https://docs.quix.io/client-library/subscribe.html
#     try:
#         stream_consumer.timeseries.on_dataframe_received = on_dataframe_received_handler
#     except KeyboardInterrupt:
#         sys.exit()

# def shutdown():
#     # close the consumer
#     topic_consumer.close()
#     sys.exit()

# token = CancellationToken()
# # subscribe to new streams being received

# print("Listening to streams. Press CTRL-C to exit.")
    
# topic_consumer.on_stream_received = on_stream_received_handler

# # Handle termination signals and provide a graceful exit
# qx.App.run(before_shutdown=shutdown)