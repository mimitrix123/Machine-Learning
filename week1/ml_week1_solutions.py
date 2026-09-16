"""Week 1 ML practice set solutions.

Questions covered:
1. Load a CSV using pandas and print first 10 rows.
2. Split dataset into train/test using sklearn.
3. Train a Linear Regression model and evaluate it.
4. Predict house price from area input.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "house_prices.csv"


def load_and_explore(path: Path) -> pd.DataFrame:
    """Q1: Load a CSV and display basic information."""
    df = pd.read_csv(path)

    print("\n--- First 10 rows ---")
    print(df.head(10))

    print("\n--- Dataset information ---")
    print(df.info())

    print("\n--- Basic statistics ---")
    print(df.describe())

    return df


def train_model(df: pd.DataFrame):
    """Q2 + Q3: split data and train/evaluate Linear Regression."""
    X = df[["area"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n--- Linear Regression results ---")
    print(f"Coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    print("\nNote: regression models should normally be evaluated with MAE/RMSE/R² rather than classification accuracy.")

    return model


def predict_house_price(model: LinearRegression) -> None:
    """Q4: Predict a house price for an area entered by the user."""
    raw_area = input("\nEnter house area in square feet: ").strip()

    try:
        area = float(raw_area)
        if area <= 0:
            raise ValueError("Area must be greater than zero.")

        prediction = model.predict(pd.DataFrame({"area": [area]}))[0]
        print(f"Predicted price: ${prediction:,.2f}")
    except ValueError as exc:
        print(f"Invalid input: {exc}")


def main() -> None:
    df = load_and_explore(DATA_PATH)
    model = train_model(df)
    predict_house_price(model)


if __name__ == "__main__":
    main()
