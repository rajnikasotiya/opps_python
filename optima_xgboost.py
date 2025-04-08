import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

# 1. Load Data
train = pd.read_csv('/mnt/data/train.csv')
test = pd.read_csv('/mnt/data/test.csv')

# 2. Correlation with Premium_Amount
numerical_data = train.select_dtypes(include=[np.number])
correlation = numerical_data.corr()['Premium_Amount'].drop('Premium_Amount')
top_5_features = correlation.abs().sort_values(ascending=False).head(5).index.tolist()

# 3. Handle missing values in top 5 features
for col in top_5_features:
    if train[col].skew() > 1 or train[col].skew() < -1:
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(test[col].median(), inplace=True)
    else:
        train[col].fillna(train[col].mean(), inplace=True)
        test[col].fillna(test[col].mean(), inplace=True)

# 4. Standardization
scaler = StandardScaler()
X = scaler.fit_transform(train[top_5_features])
X_test = scaler.transform(test[top_5_features])
y = train['Premium_Amount']

# 5. Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define Optuna optimization function
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
    }

    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)  # RMSE

# 7. Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)

# 8. Final Model with Best Parameters
best_model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X, y)

# 9. Feature Importance Plot
xgb.plot_importance(best_model, importance_type='gain', max_num_features=5)
plt.title("Top 5 Feature Importances")
plt.show()

# 10. Predict on test set
predictions = best_model.predict(X_test)

# 11. Prepare submission
submission = pd.DataFrame({'id': test['id'], 'Premium_Amount': predictions})
submission.to_csv('submission.csv', index=False)
