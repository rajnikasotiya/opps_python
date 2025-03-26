# Import necessary libraries
import optuna  # For hyperparameter tuning
import xgboost as xgb  # XGBoost model
import pandas as pd  # Data manipulation
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.metrics import accuracy_score  # Model evaluation
from sklearn.preprocessing import LabelEncoder  # Encoding categorical variables
import torch  # For GPU acceleration

# ✅ Load the Brain Cancer Dataset
# Replace 'brain_cancer.csv' with the path to your dataset
data = pd.read_csv('brain_cancer.csv')

# ✅ Preprocessing the dataset
# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()

# Assuming columns 'gender', 'location', and 'grade' are categorical
data['gender'] = label_encoder.fit_transform(data['gender'])
data['location'] = label_encoder.fit_transform(data['location'])
data['grade'] = label_encoder.fit_transform(data['grade'])

# Separate features (X) and target (y)
X = data.drop('survival_time', axis=1)  # Features
y = data['survival_time']  # Target variable

# ✅ Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define the Optuna objective function
def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Define hyperparameter search space
    params = {
        "objective": "reg:squarederror",            # Regression task
        "eval_metric": "rmse",                      # Root Mean Squared Error
        "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",  # Use GPU if available
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),    # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 12),                  # Max depth of trees
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),        # Number of boosting rounds
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),    # Min child weight
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),             # Subsample ratio
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Feature sampling ratio
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),             # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)            # L2 regularization
    }

    # ✅ Train the XGBoost model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # ✅ Make predictions
    preds = model.predict(X_test)

    # ✅ Calculate RMSE (Root Mean Squared Error)
    rmse = ((preds - y_test) ** 2).mean() ** 0.5

    # Return RMSE as the objective to minimize
    return rmse


# ✅ Use Optuna for Hyperparameter Tuning
study = optuna.create_study(direction="minimize")  # Minimize RMSE
study.optimize(objective, n_trials=50)  # Run 50 trials

# ✅ Display the best hyperparameters and score
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

# Calculate final RMSE
final_rmse = ((final_preds - y_test) ** 2).mean() ** 0.5
print(f"\n✅ Final Model RMSE: {final_rmse:.4f}")
