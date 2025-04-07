import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



train = pd.read_csv("/mnt/data/train.csv")
test = pd.read_csv("/mnt/data/test.csv")


numerical_data = train.select_dtypes(include=['int64', 'float64'])



correlation = numerical_data.corr()['Premium_Amount'].drop('Premium_Amount')
top_5_features = correlation.abs().sort_values(ascending=False).head(5).index.tolist()



for col in top_5_features:
    if train[col].skew() > 1 or train[col].skew() < -1:
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(test[col].median(), inplace=True)
    else:
        train[col].fillna(train[col].mean(), inplace=True)
        test[col].fillna(test[col].mean(), inplace=True)



scaler = StandardScaler()
X_train = scaler.fit_transform(train[top_5_features])
X_test = scaler.transform(test[top_5_features])
y_train = train['Premium_Amount']




model = LinearRegression()
model.fit(X_train, y_train)



predictions = model.predict(X_test)
