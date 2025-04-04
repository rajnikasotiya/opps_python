import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load datasets
train_file = "/mnt/data/train.csv"
test_file = "/mnt/data/test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Preview data
print(train.head())
print(train.info())
print(test.head())
print(test.info())




# Identify numerical features
num_features = train.select_dtypes(exclude='object').columns.tolist()
num_features.remove("Premium_Amount")  # Remove target variable

# Compute correlation with target
correlation = train[num_features].corr()['Premium_Amount'].abs().sort_values(ascending=False)

# Select top 10 most correlated features
top_10_features = correlation.index[:10].tolist()
print("Top 10 Features:", top_10_features)




# Identify categorical and numerical features
cat_features = train.select_dtypes(include='object').columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy='median')  # Use median for numerical features
cat_imputer = SimpleImputer(strategy='most_frequent')  # Use most common value for categorical features

train[num_features] = num_imputer.fit_transform(train[num_features])
train[cat_features] = cat_imputer.fit_transform(train[cat_features])

test[num_features] = num_imputer.transform(test[num_features])
test[cat_features] = cat_imputer.transform(test[cat_features])




# Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

train_encoded = encoder.fit_transform(train[cat_features])
test_encoded = encoder.transform(test[cat_features])

# Convert encoded data to DataFrame
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(cat_features))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(cat_features))

# Reset index to avoid issues when merging
train_encoded_df.index = train.index
test_encoded_df.index = test.index

# Drop original categorical features and concatenate encoded ones
train.drop(columns=cat_features, inplace=True)
test.drop(columns=cat_features, inplace=True)

train = pd.concat([train, train_encoded_df], axis=1)
test = pd.concat([test, test_encoded_df], axis=1)

# Standardize numerical features
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])




# Select only top 10 features + target variable
X = train[top_10_features]
y = train["Premium_Amount"]
X_test = test[top_10_features]

# Split into train-validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate model
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error on Validation Set: {mae}")

# Predict on test set
test_predictions = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({'id': test['id'], 'Premium_Amount': test_predictions})
submission.to_csv('/mnt/data/submission.csv', index=False)







#Updated Approach Without One-Hot Encoding


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load datasets
train_file = "/mnt/data/train.csv"
test_file = "/mnt/data/test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Identify numerical and categorical features
num_features = train.select_dtypes(exclude='object').columns.tolist()
cat_features = train.select_dtypes(include='object').columns.tolist()

# Remove target variable from numerical features
num_features.remove("Premium_Amount")


# Fill missing numerical values with median
num_imputer = SimpleImputer(strategy='median')
train[num_features] = num_imputer.fit_transform(train[num_features])
test[num_features] = num_imputer.transform(test[num_features])

# Fill missing categorical values with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
train[cat_features] = cat_imputer.fit_transform(train[cat_features])
test[cat_features] = cat_imputer.transform(test[cat_features])



# Apply Label Encoding
label_encoders = {}  # Store encoders for later use

for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])  # Fit and transform on train
    test[col] = le.transform(test[col])  # Transform test using same encoding
    label_encoders[col] = le  # Save encoder for future use




# Compute correlation with target variable
correlation = train.corr()['Premium_Amount'].abs().sort_values(ascending=False)

# Select top 10 most correlated features
top_10_features = correlation.index[1:11].tolist()  # Ignore 'Premium_Amount' itself
print("Top 10 Features:", top_10_features)




scaler = StandardScaler()

# Scale numerical features
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])




# Select top 10 features + target variable
X = train[top_10_features]
y = train["Premium_Amount"]
X_test = test[top_10_features]

# Split into train-validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate model
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error on Validation Set: {mae}")

# Predict on test set
test_predictions = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({'id': test['id'], 'Premium_Amount': test_predictions})
submission.to_csv('/mnt/data/submission.csv', index=False)


