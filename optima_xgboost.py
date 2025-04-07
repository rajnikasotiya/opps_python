# Re-import required packages due to environment reset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance
import optuna
import matplotlib.pyplot as plt

# Reload datasets
train = pd.read_csv('/mnt/data/train.csv')
test = pd.read_csv('/mnt/data/test.csv')

# Select numerical features and find top 5 correlated with target
numerical_data = train.select_dtypes(exclude='object')
correlation = numerical_data.corr()['Premium_Amount'].drop('Premium_Amount')
top_5_features = correlation.abs().sort_values(ascending=False).head(5).index.tolist()

# Handle missing values using skewness
for col in top_5_features:
    if train[col].skew() > 1 or train[col].skew() < -1:
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(test[col].median(), inplace=True)
    else:
        train[col].fillna(train[col].mean(), inplace=True)
        test[col].fillna(test[col].mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(train[top_5_features])
X_test = scaler.transform(test[top_5_features])
y = train['Premium_Amount']

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }
    
    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20, verbose=False)
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse

# Optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# Train final model with best params
best_params = study.best_params
final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X, y)

# Plot feature importances
plt.figure(figsize=(8, 5))
plot_importance(final_model, importance_type='gain', show_values=False)
plt.title('Feature Importances (Gain)')
plt.tight_layout()
plt.show()


