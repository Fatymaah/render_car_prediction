import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load your cleaned dataset
df = pd.read_csv(r"C:\Users\user\Documents\car_price_prediction\cleaned_car_price_data.csv")

# Features and target
X = df[['engine', 'seats', 'mileage(km/ltr/kg)', 'fuel', 'transmission', 'max_power']]
y = df['selling_price']

# Define numeric and categorical features
numeric_features = ['seats', 'mileage(km/ltr/kg)', 'engine','max_power']
categorical_features = [ 'fuel', 'transmission']

# Numeric preprocessing: impute missing values, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing: impute missing, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Final pipeline
pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Save model
joblib.dump(pipe, "car_price_pipeline.pkl")

print("Model trained and saved as car_price_pipeline.pkl")