import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify


with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
app = Flask(__name__, static_url_path="")

#this is a decorator it is known by the '/' name...not by index( )
@app.route('/') # if you get a thing, request, then run the code and return the results.
def index():   # this is the thing it runs.....
    """Return the main page."""
    return render_template('index.html')

## this is a decorator it is known by '/predict' not by the func name.....
@app.route('/predict', methods=['GET', 'POST'])  # this is where we send the request...
def predict():
    """Return a random prediction."""
    data = request.json
    prediction = model.predict_proba([data['user_input']])
    return jsonify({'odds': round(prediction[0][1], 3)})

