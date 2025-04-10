from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/logistic_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input values from the form
#         features = [
#             float(request.form['fixed_acidity']),
#             float(request.form['volatile_acidity']),
#             float(request.form['citric_acid']),
#             float(request.form['residual_sugar']),
#             float(request.form['chlorides']),
#             float(request.form['free_sulfur_dioxide']),
#             float(request.form['total_sulfur_dioxide']),
#             float(request.form['density']),
#             float(request.form['pH']),
#             float(request.form['sulphates']),
#             float(request.form['alcohol']),
#         ]

#         # Scale and predict
#         features_scaled = scaler.transform([features])
#         prediction = model.predict(features_scaled)[0]

#         quality_map = {
#             0: 'Low Quality 🍷',
#             1: 'Medium Quality 🍷',
#             2: 'High Quality 🍷'
#         }

#         return render_template('index.html', prediction_text=quality_map[prediction])
#     except Exception as e:
#         return render_template('index.html', prediction_text=f"Error: {e}")
import pandas as pd  # Make sure this is imported

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = pd.DataFrame([features], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    quality_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    result = quality_map.get(prediction[0], 'Unknown')

    return render_template('index.html', prediction_text=f'Wine Quality: {result}')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
