from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("car_price_pipeline.pkl")

# Conversion rate INR to USD
INR_TO_USD = 83  # adjust if needed

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        engine = float(request.form['engine'])
        seats = int(request.form['seats'])
        mileage = float(request.form['mileage'])
        max_power = float(request.form['max_power'])
        fuel = request.form['fuel']
        transmission = request.form['transmission']

        # Prepare input for prediction
        input_data = pd.DataFrame([{
            'engine': engine,
            'seats': seats,
            'mileage(km/ltr/kg)': mileage,
            'max_power': max_power,
            'fuel': fuel,
            'transmission': transmission
        }])

        # Make prediction in INR
        prediction_inr = model.predict(input_data)[0]

        # Convert to USD
        prediction_usd = prediction_inr / INR_TO_USD

        return render_template('index.html',
                               prediction_text=f'Estimated Selling Price: ${prediction_usd:,.2f}')
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
