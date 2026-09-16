# Week 1 – Written Answers

## Practice Set

### 1. Load a CSV using pandas and print the first 10 rows

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head(10))
```

### 2. Split dataset into train/test using scikit-learn

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

### 3. Train a Linear Regression model and check performance

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R²:", r2_score(y_test, predictions))
```

For a regression problem, MAE, RMSE and R² are more appropriate than classification accuracy.

### 4. Predict house price based on area

```python
area = 2000
predicted_price = model.predict([[area]])
print(predicted_price[0])
```

The repository version uses a DataFrame with the feature name to avoid scikit-learn feature-name warnings.

## Assignment 1 – Dataset exploration

```python
df.info()
print(df.describe())
print(df.head())
```

- `info()` shows column names, non-null counts and data types.
- `describe()` gives statistics such as count, mean, standard deviation, minimum and quartiles for numeric columns.

## Assignment 2 – Missing data

### Mean imputation

```python
df["Age"] = df["Age"].fillna(df["Age"].mean())
```

### Median imputation

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

Median is often preferable when the feature has strong outliers or a skewed distribution.

## Assignment 3 – Encoding categorical variables

### LabelEncoder

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["Sex_Encoded"] = encoder.fit_transform(df["Sex"])
```

### OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(df[["Embarked"]])
```

For a production ML pipeline, fitting encoders only on training data (or using a `Pipeline`/`ColumnTransformer`) helps prevent data leakage.

## Mini Project – Titanic Survival Prediction Data Cleaning

The implemented pipeline:

1. Loads the Kaggle Titanic CSV when `data/titanic.csv` exists.
2. Fills missing `Age` and `Fare` with their medians.
3. Fills missing `Sex` and `Embarked` values with their modes.
4. Encodes `Sex` using `LabelEncoder`.
5. One-hot encodes `Embarked`.
6. Visualizes the age distribution.
7. Writes `outputs/titanic_cleaned.csv`.

Run:

```bash
python week1/titanic_cleaning.py
```
