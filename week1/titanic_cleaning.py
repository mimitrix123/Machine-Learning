"""Titanic Survival Prediction – Week 1 data-cleaning mini project.

Tasks:
- Clean missing data.
- Encode Sex and Embarked.
- Visualize age distribution.
- Export a cleaned CSV.

Preferred input: data/titanic.csv

When data/titanic.csv is unavailable, Seaborn's public Titanic dataset is used only as a fallback demonstration.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "titanic.csv"
OUTPUT_DIR = ROOT / "outputs"
CLEANED_PATH = OUTPUT_DIR / "titanic_cleaned.csv"
PLOT_PATH = OUTPUT_DIR / "age_distribution.png"


def load_titanic() -> pd.DataFrame:
    """Load Kaggle Titanic train.csv when available, otherwise use a public fallback."""
    if INPUT_PATH.exists():
        print(f"Loading Kaggle Titanic data from: {INPUT_PATH}")
        return pd.read_csv(INPUT_PATH)

    print("data/titanic.csv not found.")
    print("Using Seaborn Titanic dataset as a fallback demonstration.")
    df = sns.load_dataset("titanic")
    return df.rename(columns={
        "survived": "Survived", "pclass": "Pclass", "sex": "Sex",
        "age": "Age", "sibsp": "SibSp", "parch": "Parch",
        "fare": "Fare", "embarked": "Embarked",
    })


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and encode the requested categorical columns."""
    cleaned = df.copy()

    if "Age" in cleaned.columns:
        cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())
    if "Fare" in cleaned.columns:
        cleaned["Fare"] = cleaned["Fare"].fillna(cleaned["Fare"].median())

    for column in ["Sex", "Embarked"]:
        if column in cleaned.columns and cleaned[column].isna().any():
            mode = cleaned[column].mode(dropna=True)
            if not mode.empty:
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])

    if "Survived" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Survived"])

    if "Sex" in cleaned.columns:
        encoder = LabelEncoder()
        cleaned["Sex_Encoded"] = encoder.fit_transform(cleaned["Sex"].astype(str))

    if "Embarked" in cleaned.columns:
        cleaned = pd.get_dummies(cleaned, columns=["Embarked"], prefix="Embarked", dtype=int)

    return cleaned


def visualize_age(df: pd.DataFrame) -> None:
    """Create the required age-distribution plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=30, kde=True)
    plt.title("Titanic Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of passengers")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved age distribution: {PLOT_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_titanic()
    print("\n--- Original data ---")
    print(df.head())
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    cleaned = clean_data(df)
    print("\n--- Cleaned data ---")
    print(cleaned.head())
    print("\nMissing values after cleaning:")
    print(cleaned.isna().sum())

    visualize_age(cleaned)
    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Saved cleaned dataset: {CLEANED_PATH}")


if __name__ == "__main__":
    main()
