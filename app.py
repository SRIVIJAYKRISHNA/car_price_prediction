from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load dataset (Ensure it contains 'Make' and 'Model' columns)
df = pd.read_csv("D:\\Projects\\Linear Regression\\Dataset\\cars_dataset.csv")

# Load trained model and preprocessing objects
with open("model/ridge_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("model/feature_selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("model/poly_transform.pkl", "rb") as f:
    poly = pickle.load(f)

@app.route('/')
def home():
    makes = sorted(df['Make'].unique())  # Get unique car brands
    return render_template('index.html', makes=makes)

@app.route('/get_models/<make>')
def get_models(make):
    """Return all models for the selected make"""
    models = df[df['Make'].str.lower() == make.lower()]['model'].unique().tolist()
    models.sort()
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Convert numerical columns to proper type
        numeric_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
        for col in numeric_cols:
            input_data[col] = pd.to_numeric(input_data[col])

        # Apply log transform where needed
        input_data["mileage"] = np.log1p(input_data["mileage"])

        # Apply scaling to numerical features
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

        # Apply one-hot encoding to categorical features
        categorical_encoded = encoder.transform(input_data[['Make', 'model', 'transmission', 'fuelType']])
        categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(), index=input_data.index)

        # Merge Encoded Features with Scaled Numerical Features
        input_data = input_data.drop(columns=['Make', 'model', 'transmission', 'fuelType']).reset_index(drop=True)
        input_data = pd.concat([input_data, categorical_df], axis=1)

        # Apply feature selection
        input_data_selected = selector.transform(input_data)

        # Apply polynomial transformation
        input_data_poly = poly.transform(input_data_selected)

        # Make prediction
        prediction = model.predict(input_data_poly)
        predicted_price = np.expm1(prediction)[0]  # Convert back from log scale

        return jsonify({'predicted_price': f"${predicted_price:,.2f}"})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
