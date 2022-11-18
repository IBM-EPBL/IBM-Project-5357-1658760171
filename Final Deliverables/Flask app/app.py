
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


# Declare a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    if request.method == "POST":
        
        model = load_model(r'C:\karthi\Projects\Nalai Thiran\Final 1\LSTM_model.h5')

        
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
        
        # Put inputs to dataframe
        test_seq = pd.DataFrame([[sensor_2,sensor_3,sensor_4,sensor_7,sensor_8,sensor_11,sensor_12,sensor_13,sensor_15,sensor_17,sensor_20,sensor_21]])
        
        # Get prediction
        y_pred_test = model.predict(test_seq,verbose=1, batch_size=200)
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)



# Running the app
if __name__ == '__main__':
    app.run(debug = True)