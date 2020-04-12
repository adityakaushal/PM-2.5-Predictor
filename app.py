from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import json
from firebase_admin import credentials, firestore, initialize_app
import pygal
from pygal.style import BlueStyle, NeonStyle, DarkStyle, LightGreenStyle, LightColorizedStyle, RedBlueStyle, LightStyle
from fbprophet import Prophet
from datetime import datetime
app = Flask(__name__)

cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()

doc_ref = db.collection(u'nodes').document(u'005').collection(u'data').where("createdAt", ">", datetime.now().timestamp() - 604800).stream()

def load_data():
    global df, df2_datetime
    data = list()
    for doc in doc_ref:
        data.append({"DateTime": doc.id, 'pm25': doc.get('pm2_5')})

    df = pd.DataFrame(data)
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)
    df['pm25'] = df['pm25'].astype('float64')
    df = df.resample('1H').mean()
    df = df.fillna(method='ffill')
    df2_datetime = df.reset_index()

@app.route('/home', methods=['GET'])
def call_json():
    line_chart = pygal.Line()
    line_chart.x_labels = df.index
    line_chart.add('Actual Data', df['pm25'][:])
    graph = line_chart.render_data_uri()
    return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values, graph = graph)

@app.route('/', methods=['GET', 'POST'])
def home():
    username = "Aditya Kaushal"
    return render_template('index.html', username=username)

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    df1 = df
    df1 = df1.reset_index()
    df1.columns = ['ds', 'y']

    if request.method == "POST":
        periods = request.form['hrs_ahead']
        horizon = int(periods)

        prophetModel = Prophet()
        prophetModel.fit(df1)
        future = prophetModel.make_future_dataframe(freq='1H', periods=horizon)
        forecast = prophetModel.predict(future)
        predictions = forecast[['ds', 'yhat']]

        predictions = predictions.set_index('ds')
        final_pred = predictions.iloc[-horizon:]

        final_pred = final_pred.reset_index()
        final_pred.columns = ['DateTime', 'pm25']
        final_pred.set_index('DateTime', inplace = True)
        final_pred.index = pd.to_datetime(final_pred.index)
        
        final_pred_datetime = final_pred.reset_index()
        frames = [df2_datetime['DateTime'].astype(str), final_pred_datetime['DateTime'].astype(str)]
        df_xlabels = pd.concat(frames, axis = 0)
        df_index = pd.DataFrame(df_xlabels)
        df_index['DateTime'] = pd.to_datetime(df_index['DateTime'])
        df_index = df_index.set_index(['DateTime'])
        df_index = df_index.reindex(df_index.index)

        line_chart = pygal.Line(style = RedBlueStyle, xlabel_rotation = 35)
        line_chart.x_labels = final_pred.index
        #line_chart.add('Actual Data', df1['y'])
        line_chart.add('Predicted Data', final_pred['pm25'][:])
        graph = line_chart.render_data_uri()

    return render_template('predictions.html', horizon = horizon, tables=[final_pred.to_html(classes='data')], titles=final_pred.columns.values, graph = graph)

if __name__ == "__main__":
    load_data()
    app.run(host = '127.0.0.1' , port = 8080, debug=False)
