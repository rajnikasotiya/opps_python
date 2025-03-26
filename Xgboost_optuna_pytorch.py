import optuna
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch

# ✅ Load Dataset
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# ✅ Define Objective Function for Optuna
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
    }
    
    # Train the XGBoost model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Prediction and evaluation
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    
    return rmse


# ✅ Use Optuna for Hyperparameter Tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# ✅ Best Parameters and Score
best_params = study.best_params
best_score = study.best_value

print("\n🎯 Best Hyperparameters:")
print(best_params)
print(f"🏅 Best RMSE: {best_score:.4f}")

# ✅ Train Final Model with Best Parameters
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# ✅ Evaluate Final Model
final_preds = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_preds, squared=False)

print(f"\n✅ Final Model RMSE: {final_rmse:.4f}")
