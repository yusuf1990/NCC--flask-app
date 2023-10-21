from flask import Flask, render_template, request
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        if not message:
            return render_template('index.html', warning="Please enter a message before checking.")

        X = np.array([message])
        X_str = np.vectorize(str)(X)
        transformed_data = data['vectorizer'].transform(X_str)

        prediction = classifier.predict(transformed_data)[0]

        result = "This message is spam." if prediction == 'spam' else "This message is not spam."

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
