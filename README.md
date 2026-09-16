# Machine Learning – Week 1

Solutions for **ML Fundamentals + Data Preprocessing** and the Week 1 practice set.

## Covered

- Machine Learning basics: supervised, unsupervised and reinforcement learning
- Pandas dataset loading and exploration
- Missing-value handling with mean/median imputation
- Label encoding and one-hot encoding
- Feature scaling
- Train/test split
- Linear Regression
- House-price prediction
- Titanic Survival Prediction data-cleaning mini project
- Age-distribution visualization
- Exporting a cleaned dataset

## Project structure

```text
Machine-Learning/
├── README.md
├── requirements.txt
├── week1/
│   ├── ml_week1_solutions.py
│   ├── titanic_cleaning.py
│   └── answers.md
└── data/
    └── house_prices.csv
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Week 1 practice solutions

```bash
python week1/ml_week1_solutions.py
```

The script:
1. Loads `data/house_prices.csv` with Pandas and prints the first 10 rows.
2. Splits the data into training and testing sets.
3. Trains a Linear Regression model and reports MAE, RMSE and R².
4. Predicts a house price from an area entered by the user.

## Run the Titanic mini project

Put the Kaggle Titanic `train.csv` file at:

```text
data/titanic.csv
```

Then run:

```bash
python week1/titanic_cleaning.py
```

Outputs:

```text
outputs/titanic_cleaned.csv
outputs/age_distribution.png
```

If `data/titanic.csv` is not present, the script can use Seaborn's public Titanic dataset as a fallback for demonstrating the cleaning pipeline. The fallback is not the Kaggle file.

## Week 1 answers

### 1. What is Machine Learning?

Machine Learning is a field of AI in which systems learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed with every rule.

### 2. Types of Machine Learning

- **Supervised learning:** learns from labeled input-output examples. Examples: regression and classification.
- **Unsupervised learning:** finds structure in unlabeled data. Examples: clustering and dimensionality reduction.
- **Reinforcement learning:** learns actions through rewards and penalties while interacting with an environment.

### 3. Why split train/test data?

The training set is used to learn model parameters. The test set is kept separate so we can estimate how the trained model performs on unseen data.

### 4. Why preprocess data?

Cleaning, encoding and scaling make raw data usable by ML algorithms and help prevent issues caused by missing values, categorical text, inconsistent ranges or invalid values.

### 5. Linear Regression

For one feature, linear regression models the relationship as:

`y = b0 + b1*x`

where `b0` is the intercept and `b1` is the learned coefficient.

For regression, **R²** is commonly used as a goodness-of-fit metric; classification-style accuracy is not the appropriate metric.

## Notes

The code is intentionally written as beginner-friendly, reusable Python rather than a one-off notebook.
