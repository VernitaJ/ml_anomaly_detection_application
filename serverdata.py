import sqlite3
from sqlite3 import Error


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection("serverdata.db")

def execute_query(query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

# anomalies
def create_anomaly_table():
    create_anomaly_table = """
    CREATE TABLE IF NOT EXISTS anomalies (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      time TEXT NOT NULL,
      actual REAL NOT NULL,
      predicted REAL NOT NULL
    );"""
    execute_query(create_anomaly_table)

# anomalies
def create_anomaly_windows_table():
    create_anomaly_table = """
    CREATE TABLE IF NOT EXISTS anomalies (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      time TEXT NOT NULL,
      actual REAL NOT NULL,
      predicted REAL NOT NULL,
      datapoint TEXT NOT NULL,
      window_start DATETIME NOT NULL,
      window_end DATETIME NOT NULL
    );"""
    execute_query(create_anomaly_table)

def insert_anomaly(time, actual, predicted, datapoint, window_start, window_end):
    insert_anomaly = """
    INSERT INTO
      anomalies (time, actual, predicted, datapoint, window_start, window_end)
    VALUES 
        ('{time}', {actual}, {predicted}, '{datapoint}', '{window_start}', '{window_end}');
    """
    execute_query(insert_anomaly)

# metrics
def create_metrics_table():
    create_metrics_table = """
    CREATE TABLE IF NOT EXISTS metrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset TEXT NOT NULL,
      mean REAL NOT NULL,
      median REAL NOT NULL,
      std REAL NOT NULL,
      mad REAL NOT NULL
    );"""
    execute_query(create_metrics_table)

def create_model_table():
    create_model_table = """
    CREATE TABLE IF NOT EXISTS model (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset TEXT NOT NULL,
      model TEXT NOT NULL,
      mae REAL NOT NULL,
      rmse REAL NOT NULL,
    );"""
    execute_query(create_model_table)

def create_dataset_table():
    create_dataset_table = """
    CREATE TABLE IF NOT EXISTS dataset (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset TEXT NOT NULL
    );"""
    execute_query(create_dataset_table)


