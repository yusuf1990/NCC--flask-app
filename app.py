from flask import Flask, render_template, request,jsonify
import numpy as np
import pickle
import lightgbm

app = Flask(__name__)

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
classifier = data['model']
vectorizer = data['vectorizer']

@app.route('/api', methods=['POST'])

def predict():

    message = request.json['message']

    X = np.array([message])
    X_str = np.vectorize(str)(X)
    transformed_data = data['vectorizer'].transform(X_str)

    prediction = classifier.predict(transformed_data)[0]

    result = "True" if prediction == 'spam' else "False"

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
