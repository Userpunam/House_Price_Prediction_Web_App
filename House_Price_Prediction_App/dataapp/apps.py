from django.apps import AppConfig
from flask import Flask, render_template, request
import pickle
import numpy as np


class DataappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dataapp'

app = Flask(__name__)

# Load the model
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        features = np.array([[feature1, feature2, feature3]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)