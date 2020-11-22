# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 23:16:24 2020

@author: home
"""

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('home.html', prediction_text= 'Diabetes : No')
    else:
        return render_template('home.html', prediction_text= 'Diabetes : Yes')


if __name__ == "__main__":
    app.run(debug=True)