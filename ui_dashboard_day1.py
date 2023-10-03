# Import necessary libraries
import csv
from datetime import timedelta, timezone, datetime
from itertools import count
import time
import dash
from dash import html
from dash import dcc
from dash import dash_table
from dash.exceptions import PreventUpdate
import io
import os.path
from sklearn.metrics import mean_squared_error
from utilities import test_metrics, retrain_ml_model, build_model

import logging
# Configure the logging settings
logging.basicConfig(filename='error.log', level=logging.ERROR)

# copy module in
import sys
sys.path.insert(1, '../new_models/')


from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from predictions import *
plotpoints = 1000


# import metrics
metrics = pd.read_csv('metrics.csv')
metrics = metrics[metrics['type'] == 'mean']
model_metrics = pd.read_csv('performance_metrics_orig.csv')
metrics = pd.merge(metrics, model_metrics, how="outer", on="dataset")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout with all pages
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Link('Home', href='/',style={'margin-left': '10px'}),
    dcc.Link('Notifications', href='/notifications', style={'margin-left': '10px'}),
    dcc.Link('ML Model', href='/model', style={'margin-left': '10px'}),
    dcc.Link('Velocity ML Model', href='/velocity_ml_model', style={'margin-left': '10px'}),
    dcc.Link('Torque ML Model', href='/torque_ml_model', style={'margin-left': '10px'}),
    # dcc.Link('Motor Temperature ML Model', href='/motor_temperature_ml_model', style={'margin-left': '10px'}),
    # dcc.Link('Position ML Model', href='/position_ml_model', style={'margin-left': '10px'}),
    html.Div(id='page-content'),
    dcc.Store(id='velocity-data'),
], style={'font-family': 'Arial, Helvetica, sans-serif'})

# Define the callback to render the appropriate page based on the URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/notifications':
        return notification_layout
    elif pathname == '/velocity_ml_model':
        return vel_model_layout
    elif pathname == '/torque_ml_model':
        return torque_model_layout
    # elif pathname == '/motor_temperature_ml_model':
    #     return motor_temp_model_layout
    # elif pathname == '/position_ml_model':
    #     return position_model_layout
    else:
        return home_layout



# Define the layout for the Home page
home_layout = html.Div([
    # Page heading
    html.H3('Anomaly Detection System'),
    
    # Tabs for different data streams
    dcc.Tabs(id='tabs', value='torque', children=[
        dcc.Tab(label='Velocity', value='velocity_calc'),
        dcc.Tab(label='Torque', value='torque'),
        dcc.Tab(label='Motor Temperature', value='motor_temperature'),
        dcc.Tab(label='Position', value='position')
    ]),
    
    # Graph showing actual data and anomalous points
    dcc.Graph(id='graph'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every 1 second
        n_intervals=plotpoints
    ),
], style={'font-family': 'Arial, Helvetica, sans-serif', 'font-size': '15px', 'padding': '10px', 'width': '100%'})

data = pd.DataFrame(columns=['dataset','window_start', 'window_end', 'comment', 'freq', 'exp_mean', 'window_mean', 'exp_rmse', 'window_rmse'])


# Define the layout for the Notification page
notification_layout = html.Div([
    # Page heading
    html.H3('Anomaly Notifications'),
    dash_table.DataTable(id='table', columns=[{"name": i.capitalize(), "id": i} for i in data.columns], style_table={'height': 500, 'width': 1400, 'overflowY': 'scroll'}, style_cell={'textAlign': 'left', 'width': '100px', 'height': 'auto', 'minWidth': '100px', 'maxWidth': '100px', 'whiteSpace': 'normal'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]),
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    ),

])

########################MODEL PAGE########################

# Define the model page layout
vel_model_layout = html.Div([
    html.H3('Machine Learning Models'),
    html.Div([
    html.H3('Velocity Model'),
    html.H4('Current Metrics'),
    html.P('MAPE Score: '+ str(metrics[metrics['dataset'] == 'velocity_calc']['mape'].values[0])),
    html.P('RMSE Score:' + str(metrics[metrics['dataset'] == 'velocity_calc']['rmse'].values[0])),
    
    html.Div([
        html.Div([
            html.H4('Model Metrics using New Data'),
            dcc.Upload(id='update_velocity', filename='velocity_calc', children=html.Button('Test Model using New Data', style={'background': 'lightblue','color':'black', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px'}), accept='.csv', style={'height': '60px', 'lineHeight': '60px', 'margin': '0'}, multiple=True),
            html.Div(id='velocity_output'),
        ]),
        html.Div([
            html.H4('Retrain model using new Data'),
            dcc.Upload(id='upload_data', filename='velocity_calc', children=html.Button('Retrain on new Data', style={'background': 'lightgreen','color':'black', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px'}), accept='.csv', style={'height': '60px', 'lineHeight': '60px',  'margin': '0'}, multiple=True),
            html.Div(id='data_uploaded'),
        ]),
        html.Div([
            html.H4('Create a New Model'),
            dcc.Input(id='velocity-input-sample', type='number', placeholder='Enter input sample'),
            dcc.Input(id='velocity-offset', type='number', placeholder='Enter offset'),
            html.Button('Create Model', id='velocity-create-button', n_clicks=0),
            dcc.Upload(id='data_for_build', filename='velocity_calc', children=html.Button('Select Data', style={'background': 'lightgreen','color':'black', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px'}), accept='.csv', style={'height': '60px', 'lineHeight': '60px',  'margin': '0'}, multiple=True),
            html.Div(id='velocity_create_output')
        ]),
    ], style={'display:': 'flex', 'flex-direction': 'row', 'padding': '10px', 'width': '80%'}),
    ], style={'display:': 'flex', 'flex-direction': 'row', 'border': '1px solid gray', 'padding': '10px', 'width': '40%'})
])

torque_model_layout = html.Div([
    html.H3('Machine Learning Models'),
    html.Div([
        html.H3('Torque Model'),
        html.H4('Current Metrics'),
        html.P('MAPE Score: ' + str(metrics[metrics['dataset'] == 'torque']['mape'].values[0])),
        html.P('RMSE Score: ' + str(metrics[metrics['dataset'] == 'torque']['rmse'].values[0])),
        
        html.Div([
            html.Div([
                html.H4('Model Metrics using New Data'),
                dcc.Upload(
                    id='update_torque',
                    filename='torque',
                    children=html.Button(
                        'Test Model using New Data',
                        style={
                            'background': 'lightblue',
                            'color': 'black',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'padding': '10px'
                        }
                    ),
                    accept='.csv',
                    style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
                    multiple=True
                ),
                html.Div(id='torque_output'),
            ]),
            html.Div([
                html.H4('Retrain model using new Data'),
                dcc.Upload(
                    id='torque_upload_data',
                    filename='torque',
                    children=html.Button(
                        'Retrain on new Data',
                        style={
                            'background': 'lightgreen',
                            'color': 'black',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'padding': '10px'
                        }
                    ),
                    accept='.csv',
                    style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
                    multiple=True
                ),
                html.Div(id='torque_data_uploaded'),
            ]),
            html.Div([
                html.H4('Create a New Model'),
                dcc.Input(id='torque-input-sample', type='number', placeholder='Enter input sample'),
                dcc.Input(id='torque-offset', type='number', placeholder='Enter offset'),
                html.Button('Create Model', id='torque-create-button', n_clicks=0),
                dcc.Upload(
                    id='data_for_build',
                    filename='torque',
                    children=html.Button(
                        'Select Data',
                        style={
                            'background': 'lightgreen',
                            'color': 'black',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'padding': '10px'
                        }
                    ),
                    accept='.csv',
                    style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
                    multiple=True
                ),
                html.Div(id='torque_create_output')
            ]),
        ], style={'display:': 'flex', 'flex-direction': 'row', 'padding': '10px', 'width': '80%'}),
    ], style={'display:': 'flex', 'flex-direction': 'row', 'border': '1px solid gray', 'padding': '10px', 'width': '40%'})
])


# motor_temp_model_layout = html.Div([
#     html.H3('Machine Learning Models'),
#     html.Div([
#         html.H3('Motor Temperature Model'),
#         html.H4('Current Metrics'),
#         html.P('MAPE Score: ' + str(metrics[metrics['dataset'] == 'motor_temp']['mape'].values[0])),
#         html.P('RMSE Score: ' + str(metrics[metrics['dataset'] == 'motor_temp']['rmse'].values[0])),
        
#         html.Div([
#             html.Div([
#                 html.H4('Model Metrics using New Data'),
#                 dcc.Upload(
#                     id='update_motor_temp',
#                     filename='motor_temp',
#                     children=html.Button(
#                         'Test Model using New Data',
#                         style={
#                             'background': 'lightblue',
#                             'color': 'black',
#                             'borderWidth': '1px',
#                             'borderStyle': 'dashed',
#                             'borderRadius': '5px',
#                             'cursor': 'pointer',
#                             'padding': '10px'
#                         }
#                     ),
#                     accept='.csv',
#                     style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
#                     multiple=True
#                 ),
#                 html.Div(id='motor_temp_output'),
#             ]),
#             html.Div([
#                 html.H4('Retrain model using new Data'),
#                 dcc.Upload(
#                     id='upload_data',
#                     filename='motor_temp',
#                     children=html.Button(
#                         'Retrain on new Data',
#                         style={
#                             'background': 'lightgreen',
#                             'color': 'black',
#                             'borderWidth': '1px',
#                             'borderStyle': 'dashed',
#                             'borderRadius': '5px',
#                             'cursor': 'pointer',
#                             'padding': '10px'
#                         }
#                     ),
#                     accept='.csv',
#                     style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
#                     multiple=True
#                 ),
#                 html.Div(id='data_uploaded'),
#             ]),
#             html.Div([
#                 html.H4('Create a New Model'),
#                 dcc.Input(id='motor_temp-input-sample', type='number', placeholder='Enter input sample'),
#                 dcc.Input(id='motor_temp-offset', type='number', placeholder='Enter offset'),
#                 html.Button('Create Model', id='motor_temp-create-button', n_clicks=0),
#                 dcc.Upload(
#                     id='data_for_build',
#                     filename='motor_temp',
#                     children=html.Button(
#                         'Select Data',
#                         style={
#                             'background': 'lightgreen',
#                             'color': 'black',
#                             'borderWidth': '1px',
#                             'borderStyle': 'dashed',
#                             'borderRadius': '5px',
#                             'cursor': 'pointer',
#                             'padding': '10px'
#                         }
#                     ),
#                     accept='.csv',
#                     style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
#                     multiple=True
#                 ),
#                 html.Div(id='motor_temp_create_output')
#             ]),
#         ], style={'display:': 'flex', 'flex-direction': 'row', 'padding': '10px', 'width': '80%'}),
#     ], style={'display:': 'flex', 'flex-direction': 'row', 'border': '1px solid gray', 'padding': '10px', 'width': '40%'})
# ])
    

# pos_model_layout = html.Div([
#     html.H3('Machine Learning Models'),
#     html.Div([
#         html.H3('Position Model'),
#         html.H4('Current Metrics'),
#         html.P('MAPE Score: ' + str(metrics[metrics['dataset'] == 'position']['mape'].values[0])),
#         html.P('RMSE Score: ' + str(metrics[metrics['dataset'] == 'position']['rmse'].values[0])),
        
#         html.Div([
#             html.Div([
#                 html.H4('Model Metrics using New Data'),
#                 dcc.Upload(
#                     id='update_position',
#                     filename='position',
#                     children=html.Button('Test Model using New Data', style={'background': 'lightblue', 'color': 'black', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px'}),
#                     accept='.csv',
#                     style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
#                     multiple=True
#                 ),
#                 html.Div(id='position_output')
#             ]),
#             html.Div([
#                 html.H4('Retrain model using new Data'),
#                 dcc.Upload(
#                     id='upload_data',
#                     filename='position',
#                     children=html.Button('Retrain on new Data', style={'background': 'lightgreen', 'color': 'black', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px'}),
#                     accept='.csv',
#                     style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
#                     multiple=True
#                 ),
#                 html.Div(id='data_uploaded')
#             ]),
#             html.Div([
#                 html.H4('Create a New Model'),
#                 dcc.Input(id='position-input-sample', type='number', placeholder='Enter input sample'),
#                 dcc.Input(id='position-offset', type='number', placeholder='Enter offset'),
#                 html.Button('Create Model', id='position-create-button', n_clicks=0),
#                 dcc.Upload(
#                     id='data_for_build',
#                     filename='position',
#                     children=html.Button('Select Data', style={'background': 'lightgreen', 'color': 'black', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px'}),
#                     accept='.csv',
#                     style={'height': '60px', 'lineHeight': '60px', 'margin': '0'},
#                     multiple=True
#                 ),
#                 html.Div(id='position_create_output')
#             ]),
#         ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '10px', 'width': '80%'}),
#     ], style={'display': 'flex', 'flex-direction': 'row', 'border': '1px solid gray', 'padding': '10px', 'width': '40%'})
# ])

@app.callback(Output('velocity_output', 'children'), 
              Input('update_velocity', 'n_clicks'),
              Input('update_velocity', 'contents'),
              State('update_velocity', 'filename'))
def update_velocity_metrics(n_clicks, contents, filename):
    data = []
    print('content: ',filename)
    # Check if a file was uploaded
    if contents:
        # Convert the file contents to a string
        content_string = contents[0].split(',')[1]
        # Convert the string to a bytes object
        decoded = base64.b64decode(content_string)
        # Read the bytes object as a pandas dataframe
        try:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        test_metrics('velocity_calc', 20, 30, data=data)
        output = get_metrics('velocity_calc')
       
        # Update the displayed scores
        return output
    return None

def get_metrics(dataset, file='performance_metrics.csv'):
    # Get the new metrics
    metrics = pd.read_csv(file)
    metrics = metrics[metrics['dataset'] == dataset].tail(1)
    mae = metrics['mae'].values[0]
    mad = metrics['mad'].values[0]
    mape = metrics['mape'].values[0]
    rmse = metrics['rmse'].values[0]
    r2 = metrics['r2'].values[0]

    output = html.Div([
            html.P(f'Mean Absolute Error: {mae}'),
            html.P(f'Mean Absolute Deviation: {mad}'),
            html.P(f'Mean Absolute Percentage Error: {mape}'),
            html.P(f'Root Mean Square Error: {rmse}'),
            html.P(f'R2 Score: {r2}')], style={'display': 'inline-block'})
    
    return output

@app.callback(Output('torque_output', 'children'),
              Input('update_torque', 'n_clicks'),
              Input('update_torque', 'contents'),
              State('update_torque', 'filename'))
def update_torque_metrics(n_clicks, contents, filename):
    data = []
    print('content', contents, filename)
    # Check if a file was uploaded
    if contents:
        # Convert the file contents to a string
        content_string = contents[0].split(',')[1]
        # Convert the string to a bytes object
        decoded = base64.b64decode(content_string)
        # Read the bytes object as a pandas dataframe
        try:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        test_metrics('torque', 30, 30, data=data)
        output = get_metrics('torque')
       
        # Update the displayed scores
        return output
    return None

@app.callback(Output('motor_temp_output', 'children'),
              Input('update_motor_temp', 'n_clicks'),
              Input('update_motor_temp', 'contents'),
              State('update_motor_temp', 'filename'))
def update_temp_metrics(contents, filename):
    data = []
    print('content', contents, filename)
    # Check if a file was uploaded
    if contents:
        # Convert the file contents to a string
        content_string = contents[0].split(',')[1]
        # Convert the string to a bytes object
        decoded = base64.b64decode(content_string)
        # Read the bytes object as a pandas dataframe
        try:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        test_metrics('motor_temperature', 30, 30, data=data)
        output = get_metrics('motor_temperature')
       
        # Update the displayed scores
        return output
    return None

@app.callback(Output('position_output', 'children'),
              Input('update_position', 'contents'),
              State('update_position', 'filename'))
def update_position_metrics(contents, filename):
    data = []
    print('content', contents, filename)
    # Check if a file was uploaded
    if contents:
        # Convert the file contents to a string
        content_string = contents[0].split(',')[1]
        # Convert the string to a bytes object
        decoded = base64.b64decode(content_string)
        # Read the bytes object as a pandas dataframe
        try:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        test_metrics('motor_temperature', 30, 30, data=data)
        output = get_metrics('motor_temperature')
        
        # Update the displayed scores
        return output
    return None

   
import base64

# Define the callback to upload and process the data file for model retraining
@app.callback(Output('data_uploaded', 'children'),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'))
def update_ml_model(contents, filename):
    data = []
    print('model: ', filename)
    model = filename
    # Check if a file was uploaded
    if contents:
        # Convert the file contents to a string
        content_string = contents[0].split(',')[1]
        # Convert the string to a bytes object
        decoded = base64.b64decode(content_string)
        # Read the bytes object as a pandas dataframe
        try:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
        print('retraining model')
        retrain_ml_model(data, "velocity", 20, 30)

        output = [(html.Div([html.H4('Model retrained with new data.')]))]

        metrics = get_metrics('velocity_calc-current', './dir/models_retrained/comparison_metrics.csv')
        new_metrics = get_metrics('velocity_calc-retrained', './dir/models_retrained/comparison_metrics.csv')

        output.append(html.Div([html.H4('Current Metrics')]))
        output.append(metrics)
        output.append(html.Div([html.H4('New Metrics')]))
        output.append(new_metrics)
        
        # Use the data to retrain the machine learning model
        # ...

        return output
    

# Define the callback to upload and process the data file for model retraining
@app.callback(Output('torque_data_uploaded', 'children'),
              Input('torque_upload_data', 'contents'),
              State('torque_upload_data', 'filename'))
def update_ml_model(contents, filename):
    data = []
    # Check if a file was uploaded
    if contents:
        # Convert the file contents to a string
        content_string = contents[0].split(',')[1]
        # Convert the string to a bytes object
        decoded = base64.b64decode(content_string)
        # Read the bytes object as a pandas dataframe
        try:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
        print('retraining model')
        retrain_ml_model(data, "torque", 30, 30)

    # check if the last line in the file has a date which is from the last 5 minutes.
    metrics = pd.read_csv('./dir/models_retrained/comparison_metrics.csv')
    metrics = metrics[metrics['dataset'] == 'torque-retrained'].tail(1)
    time_added = metrics['time_added']
    # get the minutes from the timestamp
    time_added = time_added.values[0]

    print(time_added)
    event_time = datetime.datetime.strptime(time_added, '%Y-%m-%d %H:%M:%S')
    # check if the event time is within the last 5 minutes
    if (datetime.datetime.now() - event_time).total_seconds() < 300:
        output = [(html.Div([html.H4('Model retrained with new data.')]))]

        metrics = get_metrics('torque-current', './dir/models_retrained/comparison_metrics.csv')
        new_metrics = get_metrics('torque-retrained', './dir/models_retrained/comparison_metrics.csv')

        output.append(html.Div([html.H4('Current Metrics')]))
        output.append(metrics)
        output.append(html.Div([html.H4('New Metrics')]))
        output.append(new_metrics)
        
        return output

#Define the callback to upload and process the data file for model retraining
@app.callback(Output('velocity_create_output', 'children'),
              Input('velocity-create-button', 'n_clicks'),
              [State('velocity-input-sample', 'value'),
               State('velocity-offset', 'value'),
              State('data_for_build', 'contents'), State('data_for_build', 'filename')])
def create_ml_model(n_clicks, input_sample, timestep, contents, filename):
    print('content uploaded', filename, input_sample, timestep)
    # Check if a file was uploaded
    if contents:
        df = pd.DataFrame()
        for i in range(len(filename)):
            # Convert multiple files contents to a string
            content_string = contents[i].split(',')[1]

            # Convert the string to a bytes object
            decoded = base64.b64decode(content_string)
            # Read the bytes object as a pandas dataframe
            try:
                df = pd.concat([df, pd.read_csv(io.StringIO(decoded.decode('utf-8')))], ignore_index=True)
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        print('creating new model')
        build_model('velocity_calc', input_sample, timestep, data=df)

        output = [(html.Div([html.H4('New model created with new data.')]))]
        
        # Use the data to retrain the machine learning model
        # ...

        return output


#Define the callback to upload and process the data file for model retraining
@app.callback(Output('torque_create_output', 'children'),
              Input('torque-create-button', 'n_clicks'),
              [State('torque-input-sample', 'value'),
               State('torque-offset', 'value'),
              State('data_for_build', 'contents'), State('data_for_build', 'filename')])
def create_ml_model(n_clicks, input_sample, timestep, contents, filename):
    print('content uploaded', filename, input_sample, timestep)
    # Check if a file was uploaded
    if contents:
        df = pd.DataFrame()
        for i in range(len(filename)):
            # Convert multiple files contents to a string
            content_string = contents[i].split(',')[1]

            # Convert the string to a bytes object
            decoded = base64.b64decode(content_string)
            # Read the bytes object as a pandas dataframe
            try:
                df = pd.concat([df, pd.read_csv(io.StringIO(decoded.decode('utf-8')))], ignore_index=True)
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        print('creating new model')
        build_model('torque', input_sample, timestep, data=df)

        output = [(html.Div([html.H4('New torque model created with uploaded data.')]))]
        
        # Use the data to retrain the machine learning model
        # ...

        return output

#Define the callback to upload and process the data file for model retraining
@app.callback(Output('motor_temp_create_output', 'children'),
              Input('motor_temp-create-button', 'n_clicks'),
              [State('motor_temp-input-sample', 'value'),
               State('motor_temp-offset', 'value'),
              State('data_for_build', 'contents'), State('data_for_build', 'filename')])
def create_ml_model(n_clicks, input_sample, timestep, contents, filename):
    print('content uploaded', filename, input_sample, timestep)
    # Check if a file was uploaded
    if contents:
        df = pd.DataFrame()
        for i in range(len(filename)):
            # Convert multiple files contents to a string
            content_string = contents[i].split(',')[1]

            # Convert the string to a bytes object
            decoded = base64.b64decode(content_string)
            # Read the bytes object as a pandas dataframe
            try:
                df = pd.concat([df, pd.read_csv(io.StringIO(decoded.decode('utf-8')))], ignore_index=True)
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        print('creating new model for motor_temperature model')
        build_model('motor_temperature', input_sample, timestep, data=df)

        output = [(html.Div([html.H4('New motor_temperature model created with uploaded data.')]))]
        
        # Use the data to retrain the machine learning model
        # ...

        return output
    
#Define the callback to upload and process the data file for model retraining
@app.callback(Output('position_create_output', 'children'),
              Input('position-create-button', 'n_clicks'),
              [State('position-input-sample', 'value'),
               State('position-offset', 'value'),
              State('data_for_build', 'contents'), State('data_for_build', 'filename')])
def create_ml_model(n_clicks, input_sample, timestep, contents, filename):
    print('content uploaded', filename, input_sample, timestep)
    # Check if a file was uploaded
    if contents:
        df = pd.DataFrame()
        for i in range(len(filename)):
            # Convert multiple files contents to a string
            content_string = contents[i].split(',')[1]

            # Convert the string to a bytes object
            decoded = base64.b64decode(content_string)
            # Read the bytes object as a pandas dataframe
            try:
                df = pd.concat([df, pd.read_csv(io.StringIO(decoded.decode('utf-8')))], ignore_index=True)
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        print('creating new model for position model')
        build_model('position', input_sample, timestep, data=df)

        output = [(html.Div([html.H4('New position model created with uploaded data.')]))]
        
        # Use the data to retrain the machine learning model
        # ...

        return output
    

##########################################END OF MODEL PAGE##############################################

    

# Define the callback to update the graph based on the selected data stream
@app.callback(Output('graph', 'figure'),
              [Input('interval-component', 'n_intervals'), Input('tabs', 'value')], 
              [State('graph', 'figure')])

def update(n, tab, graph):
    # if not graph:
    # read last n values from the selected dataset
    data = pd.read_csv('./day1/' + tab + '_streamed.csv')
    metrics_dataset = metrics[metrics['dataset'] == tab]
    data.columns = [tab, 'time']
    data['time'] = pd.to_datetime(data['time'])
    # get the timenow
    timenow = pd.Timestamp.now()

    current_amount = (n-plotpoints) if (n > plotpoints) else 0
    datatoshow = data[current_amount:(n+1)]
    return update_graph(tab, datatoshow)

def update_graph(tab, datatoshow):
    predictions = pd.read_csv('./day1/predictions_'+ tab +'.csv')
    predictions.columns = ['predicted', 'time']
    predictions['time'] = pd.to_datetime(predictions['time'])
    mergedStuff = pd.merge(predictions, datatoshow, on=['time'], how='inner')
    # sort mergedStuff by time
    mergedStuff = mergedStuff.sort_values(by=['time'])
    # remove duplicates
    mergedStuff = mergedStuff.drop_duplicates(subset=['time'], keep='first')
    
    # Generate a plotly figure using the latest data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=datatoshow['time'], y=datatoshow[tab],
                        mode='lines',
                        name='actual'))
    
    fig.add_trace(go.Scatter(x=mergedStuff['time'], y=mergedStuff['predicted'],
                        mode='lines',
                        name='predicted'))
    
    #4min windows
    windowsize = 60*4
    
    check_anomalies_windows(mergedStuff.tail(windowsize), tab)

    # add the anomaly points
    # update_dataset_anomalies(mergedStuff.tail(1), tab, np.mean(abs(mergedStuff['predicted'] - mergedStuff[tab])))

    # #change this to fetch from the database
    anomalies = pd.read_csv('anomalies.csv')
    anomalies = anomalies[anomalies['dataset'] == tab]
    anomalies['window_end'] = pd.to_datetime(anomalies['window_end'])
    # read last line
    anomalies = anomalies.tail(1)

    # Compare the window_end of anomalies to the time of mergedstuff
    if not anomalies.empty and (anomalies['window_end'].values[0] > mergedStuff['time'][0]):
        fig.add_vrect(x0=anomalies['window_start'].values[0], x1=np.array(anomalies['window_end'].values[0]), fillcolor="salmon", opacity=0.5, layer="below" ,line_width=2)

    # if not anomalies.empty:
    #     # fig.add_trace(go.Scatter(x=anomalies['window_start'], y=[23.3],
    #     #                 mode='markers',
    #     #                 name='anomalies'))
    #     px.scatter(x=[anomalies['window_end']], y=[42], size=[20], color=['pink'])
    
    return fig

def check_anomalies_windows(window, tab):
    start_time = time.time()

    data = metrics[metrics['dataset'] == tab]
    mean = float(data['mean'].values[0])
    rmse = float(data['rmse'].values[0])
    freq = data['freq'].values[0]
    freq = np.array(freq.strip('][').split(', '))
    freq = float(freq[3])

    if window.shape[0] < 60*4:
        return
    # get the frequency of the window, as well as the amount of points which fall within the correct frequenct range
    top_freq, count = get_frequency(window[tab])

    # check if frequency is within 0.1 of the top frequency
    if not ((top_freq >= (freq - 0.01)) and (top_freq <= (freq + 0.01)) and (count >= window.shape[0]*0.104)):
        # get the mean of the window
        window_mean = np.mean(window[tab])
        # get the mean of the actual vs predicted
        residual_mean = np.mean(abs(window['predicted'] - window[tab]))
        
        # get the metrics for the dataset
        window_rmse = np.sqrt(mean_squared_error(window['predicted'], window[tab]))

        # get percentage differences
        diff = abs(mean - window_mean)
        percentage_diff_mean = (diff/mean)*100
        rmse_test_diff = abs(window_rmse - rmse)
        rmse_test_percentage_diff = (rmse_test_diff/rmse)*100
        comment = None
        # check whether the mean of the window is greater than the mean of the dataset
        if window_mean > mean and window_rmse > rmse:
            comment = 'Actual window mean is ' + str(percentage_diff_mean) + '% greater than expected and is on average '+ str(residual_mean) +' from the predicted value. RMSE is ' + str(rmse_test_percentage_diff) + '% greater than expected.'
        elif percentage_diff_mean > 5 and rmse_test_percentage_diff < 10:
            comment = 'Actual window mean is ' + str(percentage_diff_mean) + '% greater than normal, and on average is ' + str(residual_mean) + ' from the predicted value, but the RMSE is normal.'
        elif rmse_test_diff > mean/100 and abs(window_mean - mean) < mean/100:
            comment = 'Actual window mean is normal, but the RMSE is ' + str(rmse_test_percentage_diff) + '% greater than expected.'
        else:
            comment = 'Actual window mean and RMSE are normal, but frequency is not as expected.'
        
        update_anomaly_window(window, tab, top_freq, mean, window_mean, rmse, window_rmse, comment)

        end_time = time.time()

        execution_time = end_time - start_time
        
        print(f"Time to find and update anomaly: {execution_time} seconds")


def get_frequency(window):
    window = window - window.mean()

    # get the frequency height and hz
    fft = np.fft.fft(window)
    freqs = np.fft.fftfreq(len(window))
    
    # remove the negative frequencies
    freq = freqs[:len(window)//2]
    y = np.abs(fft[:len(window)//2])
    
    # get the most frequent frequency
    most_freq = freq[np.argmax(np.abs(y))]
    
    range_low = most_freq - 0.05
    range_high = most_freq + 0.05

    freq_in_range = freq[(freq >= range_low) & (freq <= range_high)]
    return most_freq, len(freq_in_range)

def update_anomaly_window(window, tab, top_freq, exp_mean, window_mean, exp_rmse, window_rmse, comment):
    window_start = window.head(1)['time'].values[0]
    window_end = window.tail(1)['time'].values[0]
    anomaly = pd.DataFrame([{'dataset': tab, 'window_start': window_start, 'window_end': window_end,'comment':comment, 'freq' : top_freq, 'exp_mean': exp_mean, 'window_mean': window_mean, 'exp_rmse': exp_rmse, 'window_rmse': window_rmse}])
    anomaly.to_csv('anomalies.csv', mode='a', index=False, header=False)

# Define the callback to update the anomaly table based on the data in the database
@app.callback(Output('table', 'data'),
              [Input('interval-component', 'n_intervals'),])
def update_anomaly_table(n):
    
    start_time = time.time()
    
    #create csv if doesn't exist
    filename = 'anomalies.csv'
    file_exists = os.path.isfile(filename)
    if not file_exists:
        with open(filename, 'w') as csvfile:
            headers = ['dataset', 'window_start', 'window_end', 'comment', 'freq', 'exp_mean', 'window_mean', 'exp_rmse', 'window_rmse']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
            writer.writeheader()
    # add data to the csv file

    # get the last 30 minutes of data
    data = pd.read_csv('anomalies.csv')
    data['window_start'] = pd.to_datetime(data['window_start'])
    data['window_end'] = pd.to_datetime(data['window_end'])
    lasttime = data.tail(1)['window_end'].values[0]
    data = data[data['window_start'] > (lasttime - pd.Timedelta(minutes=4))]
    # data = data[data['window_rmse']> 19]

    end_time = time.time()

    
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Time to update anomaly notification list: {execution_time} seconds")

    #add the data from the other csv files to the data
    return data.to_dict('records')

# suppress callback exceptions
app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
