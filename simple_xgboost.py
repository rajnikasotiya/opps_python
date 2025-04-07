# 📦 Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 📥 Load datasets
train = pd.read_csv('/mnt/data/train.csv')
test = pd.read_csv('/mnt/data/test.csv')

# 🧹 Select only numerical features for correlation
numerical_data = train.select_dtypes(exclude='object')

# 🎯 Find top 5 correlated numerical features with the target
correlation = numerical_data.corr()['Premium_Amount'].drop('Premium_Amount')
top_5_features = correlation.abs().sort_values(ascending=False).head(5).index.tolist()
print("Top 5 Features:", top_5_features)

# 🧼 Handle missing values using mean/median based on skewness
for col in top_5_features:
    if train[col].skew() > 1 or train[col].skew() < -1:
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(test[col].median(), inplace=True)
    else:
        train[col].fillna(train[col].mean(), inplace=True)
        test[col].fillna(test[col].mean(), inplace=True)

# 🔄 Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[top_5_features])
X_test = scaler.transform(test[top_5_features])
y_train = train['Premium_Amount']

# ⚙️ Train XGBoost Regressor
xgb_model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# ✅ Predict on test set
predictions = xgb_model.predict(X_test)

# 📝 Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'Premium_Amount': predictions
})

# 💾 Save the results
submission.to_csv('xgb_submission.csv', index=False)
