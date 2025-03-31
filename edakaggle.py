linkcode
Problem statement¶
data description
Age: Age of the insured individual (Numerical)
Gender: Gender of the insured individual (Categorical: Male, Female)
Annual Income: Annual income of the insured individual (Numerical, skewed)
Marital Status: Marital status of the insured individual (Categorical: Single, Married, Divorced)
Number of Dependents: Number of dependents (Numerical, with missing values)
Education Level: Highest education level attained (Categorical: High School, Bachelor's, Master's, PhD)
Occupation: Occupation of the insured individual (Categorical: Employed, Self-Employed, Unemployed)
Health Score: A score representing the health status (Numerical, skewed)
Location: Type of location (Categorical: Urban, Suburban, Rural)
Policy Type: Type of insurance policy (Categorical: Basic, Comprehensive, Premium)
Previous Claims: Number of previous claims made (Numerical, with outliers)
Vehicle Age: Age of the vehicle insured (Numerical)
Credit Score: Credit score of the insured individual (Numerical, with missing values)
Insurance Duration: Duration of the insurance policy (Numerical, in years)
Premium Amount: Target variable representing the insurance premium amount (Numerical, skewed)
Policy Start Date: Start date of the insurance policy (Text, improperly formatted)
Customer Feedback: Short feedback comments from customers (Text)
Smoking Status: Smoking status of the insured individual (Categorical: Yes, No)
Exercise Frequency: Frequency of exercise (Categorical: Daily, Weekly, Monthly, Rarely)
Property Type: Type of property owned (Categorical: House, Apartment, Condo)
Quick overview of EDA results:
As our data is synthetic, all the missing values comes under MCAR.
5% of population's annual income is lower than premium amount paid
Insurance duration feature has inconsistency
Annual income and premium amount feature are right skewed
In Categorical features, all the categories are evenly distributed
Let's explore these results step by step with Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#additional config
pd.set_option('display.max_columns',None)
data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
data.head()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1458: RuntimeWarning: invalid value encountered in greater
  has_large_values = (abs_vals > 1e6).any()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in less
  has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in greater
  has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
id	Age	Gender	Annual Income	Marital Status	Number of Dependents	Education Level	Occupation	Health Score	Location	Policy Type	Previous Claims	Vehicle Age	Credit Score	Insurance Duration	Policy Start Date	Customer Feedback	Smoking Status	Exercise Frequency	Property Type	Premium Amount
0	0	19.0	Female	10049.0	Married	1.0	Bachelor's	Self-Employed	22.598761	Urban	Premium	2.0	17.0	372.0	5.0	2023-12-23 15:21:39.134960	Poor	No	Weekly	House	2869.0
1	1	39.0	Female	31678.0	Divorced	3.0	Master's	NaN	15.569731	Rural	Comprehensive	1.0	12.0	694.0	2.0	2023-06-12 15:21:39.111551	Average	Yes	Monthly	House	1483.0
2	2	23.0	Male	25602.0	Divorced	3.0	High School	Self-Employed	47.177549	Suburban	Premium	1.0	14.0	NaN	3.0	2023-09-30 15:21:39.221386	Good	Yes	Weekly	House	567.0
3	3	21.0	Male	141855.0	Married	2.0	Bachelor's	NaN	10.938144	Rural	Basic	1.0	0.0	367.0	1.0	2024-06-12 15:21:39.226954	Poor	Yes	Daily	Apartment	765.0
4	4	21.0	Male	39651.0	Single	1.0	Bachelor's	Self-Employed	20.376094	Rural	Premium	0.0	8.0	598.0	4.0	2021-12-01 15:21:39.252145	Poor	Yes	Weekly	House	2022.0
data.shape
(1200000, 21)
data.dtypes
id                        int64
Age                     float64
Gender                   object
Annual Income           float64
Marital Status           object
Number of Dependents    float64
Education Level          object
Occupation               object
Health Score            float64
Location                 object
Policy Type              object
Previous Claims         float64
Vehicle Age             float64
Credit Score            float64
Insurance Duration      float64
Policy Start Date        object
Customer Feedback        object
Smoking Status           object
Exercise Frequency       object
Property Type            object
Premium Amount          float64
dtype: object
features = data.columns
numerical_features = data.select_dtypes(exclude='object').columns
categorical_features = data.select_dtypes(include='object').columns
print(f'numerical features: {len(numerical_features)} \ncategorical features: {len(categorical_features)}')
numerical features: 10 
categorical features: 11
checking missing values
data.isnull().sum()
id                           0
Age                      18705
Gender                       0
Annual Income            44949
Marital Status           18529
Number of Dependents    109672
Education Level              0
Occupation              358075
Health Score             74076
Location                     0
Policy Type                  0
Previous Claims         364029
Vehicle Age                  6
Credit Score            137882
Insurance Duration           1
Policy Start Date            0
Customer Feedback        77824
Smoking Status               0
Exercise Frequency           0
Property Type                0
Premium Amount               0
dtype: int64
data['id'].nunique()
1200000
Assumption 1: as our id is unique, we assume that no two claims are overlap by an individual (i.e), we have 12 lakh indivial people who are insured in this data.

na_features = []
for col in features:
    if data[col].isnull().any():
        na_features.append(col)
        print(f'{col:<22} has {np.round(data[col].isnull().mean() * 100,2)} % of NA')
print(f'\nNo of NA features: {len(na_features)}')
print(na_features)
        
Age                    has 1.56 % of NA
Annual Income          has 3.75 % of NA
Marital Status         has 1.54 % of NA
Number of Dependents   has 9.14 % of NA
Occupation             has 29.84 % of NA
Health Score           has 6.17 % of NA
Previous Claims        has 30.34 % of NA
Vehicle Age            has 0.0 % of NA
Credit Score           has 11.49 % of NA
Insurance Duration     has 0.0 % of NA
Customer Feedback      has 6.49 % of NA

No of NA features: 11
['Age', 'Annual Income', 'Marital Status', 'Number of Dependents', 'Occupation', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration', 'Customer Feedback']
Result: 11 features have missing values

Exploratory data analysis
Analysis for feature with missing values
Age feature analysis

data['Age'].nunique()
47
data['Age'].value_counts().sort_index().plot(kind='bar',figsize=(12,3))
<Axes: xlabel='Age'>

data[data['Age'].isna()]['Gender'].value_counts()
Gender
Male      9381
Female    9324
Name: count, dtype: int64
Result: Age missing is not dependend on Gender

Analysing Occupation feature

data['Education Level'].value_counts().plot.bar()
<Axes: xlabel='Education Level'>

#checking whether educational level has any relation
data[data['Occupation'].isna()]['Education Level'].value_counts().plot(kind='pie',autopct='%1.1f%%')
<Axes: ylabel='count'>

Result: Missing values of occupation is almost equally distributed among education level. type: MCAR

Analysing annual income feature

data['Occupation'].value_counts()
Occupation
Employed         282750
Self-Employed    282645
Unemployed       276530
Name: count, dtype: int64
data[data['Annual Income'].isna()]['Occupation'].value_counts().plot(kind='pie',autopct="%1.1f%%")
<Axes: ylabel='count'>

linkcode
data[data['Annual Income'].isna()]['Education Level'].value_counts().plot(kind='pie',autopct="%1.1f%%")
<Axes: ylabel='count'>

Result: Missing values in annual income is equally distrubed among education level and occupation

Analyzing maritial status

data['Marital Status'].value_counts().plot(kind='pie',autopct='%1.1f%%')
<Axes: ylabel='count'>

data[data['Marital Status'].isna()]['Number of Dependents'].value_counts().plot(kind='pie',autopct='%1.1f%%')
<Axes: ylabel='count'>

data.groupby('Number of Dependents')['Marital Status'].value_counts().plot.bar(figsize=(6,3))
<Axes: xlabel='Number of Dependents,Marital Status'>

Result:

No of dependents has no relationship with marital status
No of dependents column is float type i.e need to change to int
Analyzing health score feature

data.columns
Index(['id', 'Age', 'Gender', 'Annual Income', 'Marital Status',
       'Number of Dependents', 'Education Level', 'Occupation', 'Health Score',
       'Location', 'Policy Type', 'Previous Claims', 'Vehicle Age',
       'Credit Score', 'Insurance Duration', 'Policy Start Date',
       'Customer Feedback', 'Smoking Status', 'Exercise Frequency',
       'Property Type', 'Premium Amount'],
      dtype='object')
data['Smoking Status'].value_counts()
Smoking Status
Yes    601873
No     598127
Name: count, dtype: int64
data.groupby('Smoking Status')['Health Score'].median().plot.bar(figsize=(6,3))
<Axes: xlabel='Smoking Status'>

#Analysing health score by excercise habit
data['Exercise Frequency'].value_counts()
data[data['Health Score'].isna()]['Exercise Frequency'].value_counts()
Exercise Frequency
Weekly     18715
Daily      18680
Rarely     18521
Monthly    18160
Name: count, dtype: int64
data.groupby('Exercise Frequency')['Health Score'].mean()
Exercise Frequency
Daily      25.650123
Monthly    25.575288
Rarely     25.629956
Weekly     25.601310
Name: Health Score, dtype: float64
Result: In this dataset, smoking status has no affect on health score. Also, Excercising doesn't have any health improvements on avg

Analysing for insurance duration

data[data['Insurance Duration'].isna()]
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1458: RuntimeWarning: invalid value encountered in greater
  has_large_values = (abs_vals > 1e6).any()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in less
  has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in greater
  has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
id	Age	Gender	Annual Income	Marital Status	Number of Dependents	Education Level	Occupation	Health Score	Location	Policy Type	Previous Claims	Vehicle Age	Credit Score	Insurance Duration	Policy Start Date	Customer Feedback	Smoking Status	Exercise Frequency	Property Type	Premium Amount
711358	711358	64.0	Male	30206.0	Married	3.0	Master's	Employed	49.551038	Suburban	Basic	0.0	18.0	581.0	NaN	2022-04-06 15:21:39.203442	Poor	Yes	Rarely	Apartment	1044.0
#checking insurance duration for similar policy start data
data[data['Policy Start Date'].str.contains('2022-04-06')].head()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1458: RuntimeWarning: invalid value encountered in greater
  has_large_values = (abs_vals > 1e6).any()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in less
  has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
/usr/local/lib/python3.10/dist-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in greater
  has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
id	Age	Gender	Annual Income	Marital Status	Number of Dependents	Education Level	Occupation	Health Score	Location	Policy Type	Previous Claims	Vehicle Age	Credit Score	Insurance Duration	Policy Start Date	Customer Feedback	Smoking Status	Exercise Frequency	Property Type	Premium Amount
2613	2613	36.0	Female	76683.0	Single	2.0	High School	NaN	5.594031	Suburban	Basic	1.0	5.0	433.0	3.0	2022-04-06 15:21:39.126661	Poor	Yes	Daily	Condo	746.0
3018	3018	31.0	Male	38323.0	Single	2.0	PhD	Employed	29.993333	Urban	Comprehensive	NaN	10.0	717.0	1.0	2022-04-06 15:21:39.246098	Good	Yes	Rarely	Condo	878.0
3277	3277	19.0	Female	38962.0	Single	4.0	Master's	NaN	13.849282	Suburban	Premium	0.0	1.0	597.0	2.0	2022-04-06 15:21:39.191219	Average	No	Weekly	Apartment	531.0
5172	5172	27.0	Male	NaN	Single	4.0	PhD	Self-Employed	29.639388	Rural	Comprehensive	NaN	10.0	705.0	9.0	2022-04-06 15:21:39.182597	NaN	No	Rarely	Apartment	24.0
5645	5645	18.0	Male	107964.0	Divorced	3.0	PhD	NaN	26.766173	Urban	Basic	1.0	6.0	432.0	6.0	2022-04-06 15:21:39.177724	Poor	Yes	Rarely	Condo	439.0
Result: For same policy start date, there are different insurance duration (inconsistency).

Further analysis

data[data['Premium Amount'] > data['Annual Income']].shape
(51717, 21)
result: around 4% people's annual income is lower than premium amount they paid

Distribution of numerical features
for col in numerical_features:
    data[col].plot(kind='hist',bins=20,figsize=(6,3),edgecolor='black')
    plt.title(col)
    plt.show()










Insights:

Annual income and premium amount are right skewed
Distribution of categorical features:
for col in categorical_features:
    if col != 'Policy Start Date':
        data[col].value_counts().plot(kind='bar',figsize=(6,3))
        plt.title(col)
        plt.show()










No Insights

Outliers Detection Analysis
Starting with numerical features

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
for col in numerical_features:
    if data[col].skew() > 0.5 or data[col].skew() < -0.5:
        print(f'{col} {np.round(data[col].skew(),2)}')
Annual Income 1.47
Previous Claims 0.91
Premium Amount 1.24
So, features with outliers:

Annual Income
Previous claims
Premium amount
Analysing Annual income:

sns.distplot(data['Annual Income'])
<Axes: xlabel='Annual Income', ylabel='Density'>

sns.boxplot(data['Annual Income'])
<Axes: >

Lets go with IQR to find extreme outliers and treat them

IQR = data['Annual Income'].quantile(0.75) - data['Annual Income'].quantile(0.25)
lower_bound = (data['Annual Income'].quantile(0.25)) - (IQR * 1.5)
upper_bound = (data['Annual Income'].quantile(0.75)) + (IQR * 1.5)
print(lower_bound,upper_bound)
-46948.5 99583.5
data[data['Annual Income'] > 99583].shape
(67132, 21)
#for extreme outliers
lower_bound = (data['Annual Income'].quantile(0.25)) - (IQR * 3)
upper_bound = (data['Annual Income'].quantile(0.75)) + (IQR * 3)
print(lower_bound,upper_bound)
-101898.0 154533.0
data[data['Annual Income'] > 99583].shape
(67132, 21)
Result: Annual Income has 67132 outliers

Analying previous claims

sns.distplot(data['Previous Claims'])
<Axes: xlabel='Previous Claims', ylabel='Density'>

sns.boxplot(data['Previous Claims'])
<Axes: >

IQR = data['Previous Claims'].quantile(0.75) - data['Previous Claims'].quantile(0.25)
lower_bound = (data['Previous Claims'].quantile(0.25)) - (IQR * 1.5)
upper_bound = (data['Previous Claims'].quantile(0.75)) + (IQR * 1.5)
print(lower_bound,upper_bound)
-3.0 5.0
data[data['Previous Claims'] > 5].shape
(369, 21)
Result: Previous claims has 369 outliers

Analyzing Premium amount

sns.distplot(data['Premium Amount'])
<Axes: xlabel='Premium Amount', ylabel='Density'>

sns.boxplot(data['Premium Amount'])
<Axes: >

IQR = data['Premium Amount'].quantile(0.75) - data['Premium Amount'].quantile(0.25)
lower_bound = (data['Premium Amount'].quantile(0.25)) - (IQR * 1.5)
upper_bound = (data['Premium Amount'].quantile(0.75)) + (IQR * 1.5)
print(lower_bound,upper_bound)
-978.5 3001.5
data[data['Premium Amount'] > 3001.5].shape
(49320, 21)
Result: Premium amount has 49320 outliers

for Categorical features

we check for rare categories and mark it as others

for col in categorical_features:
    if col != 'Policy Start Date':
        data[col].value_counts().plot(kind='pie',autopct="%1.1f%%",figsize=(3,3))
        plt.title(col)
        plt.show()










Result: In our categorical features, all the categories are evenly distributed. So let's leave it
