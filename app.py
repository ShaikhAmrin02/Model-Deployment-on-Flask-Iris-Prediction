from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('iris_model.pkl')

# Mapping of numeric predictions to species names
species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df = df.astype(float)
    prediction = model.predict(df)
    species_name = species[int(prediction)]  # Convert NumPy array to scalar
    return render_template('index.html', prediction=species_name)

if __name__ == '__main__':
    app.run(debug=True)

