import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, render_template
import pandas as pd
import joblib
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense , LSTM, Dropout
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import tensorflow as tf
import os
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
  if request.method == "POST":

    sensor_2 = request.form.get("sensor_2")
    sensor_3 = request.form.get("sensor_3")
    sensor_4 = request.form.get("sensor_4")
    sensor_7 = request.form.get("sensor_7")
    sensor_8 = request.form.get("sensor_8")
    sensor_11 = request.form.get("sensor_11")
    sensor_12 = request.form.get("sensor_12")
    sensor_13 = request.form.get("sensor_13")
    sensor_15 = request.form.get("sensor_15")
    sensor_17 = request.form.get("sensor_17")
    sensor_20 = request.form.get("sensor_20")
    sensor_21 = request.form.get("sensor_21")
    cycles = request.form.get("cycles")
    cycles = int(cycles)

    if cycles < 50:
      prediction = "Fly over the sky, long way to go!!! Lot of cycles remainding"
    else:
      json_file = open('model.json', 'r')
      model_json = json_file.read()
      model = model_from_json(model_json)
      model.load_weights("model.h5")
      test_df_1 = pd.read_csv('test_d.csv',index_col=0)
      test_df_1 = test_df_1[test_df_1['id'] == 20]
      test_df_1 = test_df_1[test_df_1['cycles']<cycles]
        # Put inputs to dataframe
      test_df_1 = test_df_1.drop(['id'],axis=1)
      test_seq = pd.DataFrame([[cycles,sensor_2,sensor_3,sensor_4,sensor_7,sensor_8,sensor_11,sensor_12,sensor_13,sensor_15,sensor_17,sensor_20,sensor_21]], columns = ['cycles','sensor_2','sensor_3','sensor_4','sensor_7','sensor_8','sensor_11','sensor_12','sensor_13','sensor_15','sensor_17','sensor_20','sensor_21'])
      test_df_1 = pd.concat([test_df_1,test_seq])
      test_df_1 = test_df_1.drop(['cycles'],axis=1)
      test_df_1 = test_df_1.tail(50)
      min_max_scaler = MinMaxScaler()
      test_df_1 = pd.DataFrame(min_max_scaler.fit_transform(test_df_1))
      test_df_1 = np.asarray(test_df_1).astype(np.float32)
      test_df_1 = [test_df_1]
      test_seq_1 = np.asarray(test_df_1).astype(np.float32)
      np.shape(test_seq_1)
        # Get prediction
      predict = model.predict(test_seq_1,verbose=1, batch_size=200)
      predict = predict-20

      if predict <=20:
        prediction = "Need to change the engine, Aircraft has less than 20 cycles of lifetime"
      else:
        prediction = "Fly over the sky, Engine has "+str(predict)+" cycles more....."
        
  else:
    prediction = ""
        
  return render_template("website.html", output = prediction)

  



if __name__ == '__main__':
    app.run(debug = True)