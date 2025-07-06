import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string # type: ignore
# --- Data Loading and Preprocessing ---
data_path = "CarsDataset.csv"
features = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'Color']
target = "selling_price"
data = pd.read_csv(data_path)
X = data[features]
y = data[target]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [col for col in features if col not in num_cols]
# Feature engineering: add squared terms
for col in num_cols:
    X[f'{col}^2'] = X[col] ** 2
# Fill missing values
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X[f'{col}^2'] = X[f'{col}^2'].fillna(X[f'{col}^2'].median())
for col in cat_cols:
    X[col] = X[col].fillna('missing')
squared_cols = [f'{col}^2' for col in num_cols]
all_num_cols = num_cols + squared_cols
# One-hot encoding
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)
feature_names = X_encoded.columns.tolist()
def align_dummies(df, feature_names):
    missing_cols = list(set(feature_names) - set(df.columns))
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_df], axis=1)
    df = df[feature_names]
    return df
def normalize(X_train, X_test):
    norm_params = {}
    for col in all_num_cols:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)
        mean = X_train[col].mean()
        std = X_train[col].std()
        norm_params[col] = (mean, std)
        X_train[col] = (X_train[col] - mean) / std if std != 0 else 0
        X_test[col] = (X_test[col] - mean) / std if std != 0 else 0
    return X_train, X_test, norm_params
def linear_regression_predict(X_train, y_train, X_test):
    X_train_np = X_train.values.astype(float)
    X_test_np = X_test.values.astype(float)
    X_train_bias = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    X_test_bias = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])
    y_train_np = y_train.values.reshape(-1, 1)
    weights = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train_np
    y_pred = X_test_bias @ weights
    return y_pred.ravel()
# Train/test split
test_size = 0.2
n = len(X_encoded)
indices = np.arange(n)
np.random.seed(42)
np.random.shuffle(indices)
split_idx = int(n * (1 - test_size))
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]
X_train = X_encoded.iloc[train_idx].reset_index(drop=True)
X_test = X_encoded.iloc[test_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)
X_train = align_dummies(X_train, feature_names)
X_test = align_dummies(X_test, feature_names)
X_train, X_test, norm_params = normalize(X_train, X_test)
X_train = X_train.astype(float)
X_test = X_test.astype(float)
# Train and evaluate
test_preds = linear_regression_predict(X_train, y_train, X_test)
mse_test = np.mean((y_test.values - test_preds) ** 2)
final_ss_res = np.sum((y_test.values - test_preds) ** 2)
final_ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
final_r_squared = 1 - (final_ss_res / final_ss_tot)
# --- HTML Template as a string ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f6f8fa; }
        .container { background: #fff; padding: 30px; border-radius: 10px; max-width: 600px; margin: auto; box-shadow: 0 4px 16px #ddd; }
        h1 { color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        .btn { background: #007bff; color: #fff; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #e9ffe9; border: 1px solid #b2f0b2; border-radius: 4px; }
        .error { margin-top: 20px; padding: 15px; background: #ffe9e9; border: 1px solid #f0b2b2; border-radius: 4px; color: #b30000; }
        .metrics { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Car Selling Price Prediction</h1>
        <form method="POST">
            {% for col in num_cols %}
            <div class="form-group">
                <label for="{{col}}">{{col.replace('_', ' ').title()}}</label>
                <input type="number" step="any" name="{{col}}" id="{{col}}" required>
            </div>
            {% endfor %}
            {% for col in cat_cols %}
            <div class="form-group">
                <label for="{{col}}">{{col.replace('_', ' ').title()}}</label>
                <input type="text" name="{{col}}" id="{{col}}" required>
            </div>
            {% endfor %}
            <button type="submit" class="btn">Predict</button>
        </form>
        {% if prediction %}
        <div class="result">
            <strong>Predicted Value:</strong> ₹{{prediction}}
        </div>
        {% endif %}
        {% if error %}
        <div class="error">
            {{error}}
        </div>
        {% endif %}
        <div class="metrics">
        </div>
    </div>
</body>
</html>
"""

# --- Flask Application ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    mse_test_disp = f"{mse_test:.2f}"
    final_r_squared_disp = f"{final_r_squared:.2f}"
    error = None

    if request.method == 'POST':
        try:
            user_input = {}
            for col in num_cols:
                val = request.form.get(col)
                val = float(val) if val and ('.' in str(val) or 'e' in str(val).lower()) else int(val)
                user_input[col] = val
                user_input[f'{col}^2'] = float(val) ** 2
            for col in cat_cols:
                val = request.form.get(col)
                user_input[col] = val if val and val.strip() != '' else 'missing'
            df = pd.DataFrame([user_input])
            for col in num_cols:
                if col not in df or pd.isnull(df[col][0]):
                    df[col] = [0]
                if f'{col}^2' not in df or pd.isnull(df[f'{col}^2'][0]):
                    df[f'{col}^2'] = [0]
            for col in cat_cols:
                if col not in df or pd.isnull(df[col][0]):
                    df[col] = ['missing']
            df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
            df_encoded = align_dummies(df_encoded, feature_names)
            for col in all_num_cols:
                if col in df_encoded:
                    mean, std = norm_params[col]
                    if std != 0:
                        df_encoded[col] = (df_encoded[col] - mean) / std
                    else:
                        df_encoded[col] = 0
            df_encoded = df_encoded.astype(float)
            predicted = linear_regression_predict(X_train, y_train, df_encoded)[0]
            prediction = f"{predicted:.2f}"
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template_string(
        HTML_TEMPLATE,
        num_cols=num_cols,
        cat_cols=cat_cols,
        prediction=prediction,
        mse_test=mse_test_disp,
        r_squared=final_r_squared_disp,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
