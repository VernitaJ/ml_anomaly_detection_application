import json
import os
import pandas as pd
import numpy as np
import joblib
import datetime

import requests

datanumber = 0

def prediction(data, dataset, model):
    scaler = None
    if os.path.exists('./dir/scalers/'+dataset+'_scaler.save'):
        scaler = joblib.load('./dir/scalers/'+dataset+'_scaler.save')
        data_scaled = scaler.transform(data[dataset].values.reshape(-1, 1))
    else: data_scaled = data[dataset].values.reshape(-1, 1)
          
    req = json.dumps({"signature_name": "serving_default", "instances": [data_scaled.tolist()]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/' + model + ':predict', data=req, headers=headers)
    print(json_response.text)
    prediction = json.loads(json_response.text)['predictions']

    prediction = np.array(prediction[0]).reshape(-1, 1)
    if scaler is not None:
        prediction = scaler.inverse_transform(prediction)
    time_entry = datetime.datetime.strptime(data['time'].iloc[-1][0:19], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=1)

    # add a prediction df
    add_prediction = pd.DataFrame([{'predicted':prediction[0][0], 'time':time_entry}])
    print('adding prediction to')
    add_prediction.to_csv('predictions_'+ dataset + '.csv', mode='a', index=False, header=False)
