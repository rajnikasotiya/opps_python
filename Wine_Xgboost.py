# ✅ Import necessary libraries
import optuna  # For hyperparameter tuning
import xgboost as xgb  # XGBoost model
import pandas as pd  # Data manipulation
from sklearn.datasets import load_wine  # Wine dataset
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.metrics import accuracy_score  # Model evaluation
import torch  # For GPU acceleration

# ✅ Load the Wine Recognition Dataset
data = load_wine()

# Convert to DataFrame for easier handling
X = pd.DataFrame(data.data, columns=data.feature_names)  # Features
y = pd.Series(data.target)  # Target labels

# ✅ Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define the Optuna objective function
def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Define hyperparameter search space
    params = {
        "objective": "multi:softmax",               # Multi-class classification
        "num_class": len(data.target_names),        # Number of classes
        "eval_metric": "mlogloss",                  # Multi-class log loss metric
        "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",  # Use GPU if available
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),    # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 12),                  # Max tree depth
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),        # Number of boosting rounds
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),    # Min child weight
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),             # Row sampling ratio
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Feature sampling ratio
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),             # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)            # L2 regularization
    }

    # ✅ Train the XGBoost model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # ✅ Make predictions
    preds = model.predict(X_test)

    # ✅ Calculate accuracy
    accuracy = accuracy_score(y_test, preds)

    # Return negative accuracy to minimize the objective
    return -accuracy


# ✅ Use Optuna for Hyperparameter Tuning
study = optuna.create_study(direction="minimize")  # Minimize negative accuracy
study.optimize(objective, n_trials=50)  # Run 50 trials

# ✅ Display the best hyperparameters and score
best_params = study.best_params
best_score = -study.best_value  # Convert negative accuracy back to positive

print("\n🎯 Best Hyperparameters:")
print(best_params)
print(f"🏅 Best Accuracy: {best_score:.4f}")

# ✅ Train Final Model with Best Parameters
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# ✅ Evaluate Final Model
final_preds = final_model.predict(X_test)

# ✅ Calculate final accuracy
final_accuracy = accuracy_score(y_test, final_preds)
print(f"\n✅ Final Model Accuracy: {final_accuracy:.4f}")
