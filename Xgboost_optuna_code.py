pip install xgboost optuna torch scikit-learn pandas



import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# ✅ Load the dataset (Iris dataset for classification)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Load the dataset into a DataFrame
df = pd.read_csv(url, header=None, names=columns)

# Map class labels to integers
df['class'] = df['class'].astype('category').cat.codes  # Encode labels as 0, 1, 2

# Split dataset into features and labels
X = df.drop('class', axis=1)
y = df['class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define the Optuna objective function
def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    params = {
        "objective": "multi:softmax",               # Multi-class classification
        "num_class": 3,                             # Number of classes in the Iris dataset
        "eval_metric": "mlogloss",                  # Multi-class log loss metric
        "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),  # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 10),               # Depth of trees
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),     # Number of boosting rounds
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10), # Min child weight
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),          # Subsample ratio
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Column sampling
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),          # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)         # L2 regularization
    }
    
    # ✅ Train the XGBoost model with the current hyperparameters
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # ✅ Make predictions
    preds = model.predict(X_test)
    
    # ✅ Calculate accuracy
    accuracy = accuracy_score(y_test, preds)
    
    # Return the negative accuracy (since Optuna minimizes the objective)
    return -accuracy


# ✅ Use Optuna for Hyperparameter Tuning
study = optuna.create_study(direction="minimize")  # Minimize negative accuracy
study.optimize(objective, n_trials=50)  # 50 trials

# ✅ Display the best parameters and accuracy
best_params = study.best_params
best_score = -study.best_value  # Converting negative accuracy back to positive

print("\n🎯 Best Hyperparameters:")
print(best_params)
print(f"🏅 Best Accuracy: {best_score:.4f}")

# ✅ Train Final Model with Best Parameters
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# ✅ Evaluate Final Model
final_preds = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)

print(f"\n✅ Final Model Accuracy: {final_accuracy:.4f}")




