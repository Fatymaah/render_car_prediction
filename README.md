# Car Price Prediction Web App

This project is a **Flask web application** that predicts the selling price of cars based on features such as engine size, mileage, seats, fuel type, transmission, and max power.

It uses **scikit-learn pipelines** with `ColumnTransformer` for preprocessing and a regression model to make predictions.

---

## 🚀 Features

* End-to-end **machine learning pipeline** with preprocessing.
* Web interface built with **Flask** and **HTML**.
* Converts predictions from INR to USD for better readability.
* Modular setup:

  * `train_model.py` → trains and saves the ML pipeline.
  * `app.py` → serves predictions with Flask.
  * `templates/index.html` → user input form & result display.

---

## 📂 Project Structure

```
├── train_model.py          # Script to train and save the ML model
├── app.py                  # Flask app to serve predictions
├── car_price_pipeline.pkl  # Saved ML pipeline (created after training)
├── templates/
│   └── index.html          # Frontend form for predictions
└── cleaned_car_price_data.csv  # Input dataset (provide your own)
```

---

## ⚙️ Installation

1. **Clone the repo** or download project files.

   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Create a virtual environment** (recommended).

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**.

   ```bash
   pip install -r requirements.txt
   ```

If you don’t have a `requirements.txt`, you can create one with:

```bash
pip freeze > requirements.txt
```

---

## 📊 Train the Model

Run the training script to create `car_price_pipeline.pkl`:

```bash
python train_model.py
```

This will:

* Load `cleaned_car_price_data.csv`.
* Preprocess numeric and categorical features.
* Train a regression model.
* Save the trained pipeline as `car_price_pipeline.pkl`.

---

## 🌐 Run the Web App

Start the Flask server:

```bash
python app.py
```

Go to your browser at:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖥️ Usage

1. Enter car details (engine size, mileage, seats, fuel, transmission, max power).
2. Click **Predict Price**.
3. Get the **estimated selling price in USD**.

---

## 🛠 Requirements

* Python 3.8+
* Flask
* scikit-learn
* pandas
* joblib

---

## 📌 Notes

* Update `INR_TO_USD` in `app.py` if exchange rates change.
* Ensure the dataset (`cleaned_car_price_data.csv`) matches the expected feature columns.

---

## 📜 License

This project is open-source. Use it, modify it, and improve it!
