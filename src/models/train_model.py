import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features.feature_engineering import engineer_features


def load_data():
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM kaggle_Loan_default", conn)
    conn.close()
    return df


def train():
    df = load_data()
    df = engineer_features(df)

    target = "Default"

    numeric_features = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm",
        "DTIRatio", "LoanIncomeRatio"
    ]

    X = df[numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Baseline Logistic Regression AUC: {auc:.4f}")


if __name__ == "__main__":
    train()
