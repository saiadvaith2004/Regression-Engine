# The Website to see the result is http://127.0.0.1:5000
from flask import Flask, request, render_template_string # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
import xgboost as xgb # type: ignore
app = Flask(__name__) #Creates an instance of the in-built class named Flask which is in the module flask
# Load the dataset
data = pd.read_csv("CarsDataset.csv")
# Preprocess and train the model
#Train Model function is called in the index() function
def train_model(modelname):
    model_data = data[data['name'] == modelname].copy()
    if model_data.empty:
        return None
    X = model_data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
    y = model_data['selling_price']
    numeric_features = ['year', 'km_driven']
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train) # The model gets trained on the given dataset
    return model_pipeline
@app.route('/', methods=['GET', 'POST'])
#The index function is called automatically when the request is made through entering the url http://127.0.0.1:5000
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        modelname = request.form['modelname']
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = request.form['fuel']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        model_pipeline = train_model(modelname)
        if model_pipeline is None:
            error = f"No data available for the model '{modelname}'."
        else:
            prediction_input = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner]
            })
            predicted_price = model_pipeline.predict(prediction_input)
            prediction = round(float(predicted_price[0]), 2)
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f0f4f8;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background: white;
            padding: 24px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            width: 360px;
        }
        h2 {
            margin-bottom: 24px;
            text-align: center;
            color: #333;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            border-color: #007BFF;
            outline: none;
        }
        button {
            width: 100%;
            padding: 14px 0;
            font-size: 16px;
            font-weight: 700;
            color: white;
            background: #007BFF;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        button:hover {
            background: #0056b3;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            font-size: 18px;
            font-weight: 700;
            color: #222;
        }
        .error {
            color: red;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Car Price Predictor</h2>
        <form method="post">
            <label for="modelname">Car Model Name</label>
            <input type="text" id="modelname" name="modelname" required />
            <label for="year">Year</label>
            <input type="number" id="year" name="year" min="1900" max="2025" required />
            <label for="km_driven">Kilometers Driven</label>
            <input type="number" id="km_driven" name="km_driven" min="0" required />
            <label for="fuel">Fuel Type</label>
            <input type="text" id="fuel" name="fuel" required />
            <label for="seller_type">Seller Type</label>
            <input type="text" id="seller_type" name="seller_type" required />
            <label for="transmission">Transmission</label>
            <input type="text" id="transmission" name="transmission" required />
            <label for="owner">Owner Type</label>
            <input type="text" id="owner" name="owner" required />
            <button type="submit">Predict Price</button>
        </form>
        {% if prediction is not none %}
            <div class="result">Predicted Price: ₹{{ prediction }}</div>
        {% elif error %}
            <div class="error">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
'''
if __name__ == '__main__':
    app.run(debug=True)