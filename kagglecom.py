import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna


# Load training and test datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')

# Inspect data
print(train.head())
print(test.head())


# Drop ID columns
train.drop('id', axis=1, inplace=True)
test_ids = test['id']
test.drop('id', axis=1, inplace=True)

# Separate features and target
X = train.drop('Premium_Amount', axis=1)
y = train['Premium_Amount']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Split the data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Train XGBoost model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse}")



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 42
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)


# Train the model with best hyperparameters
final_model = XGBRegressor(**best_params)
final_model.fit(X_scaled, y)

# Make predictions
predictions = final_model.predict(test_scaled)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'Premium_Amount': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved.")

