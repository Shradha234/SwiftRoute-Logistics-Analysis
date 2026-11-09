"""SwiftRoute Week 4 predictive modeling workflow.
Run from repository root: python src/week4_modeling.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/SwiftRoute_Logistics_Dataset.csv")
FEATURES = ["Distance_KM", "Shipment_Weight_KG", "Shipment_Volume",
            "Transportation_Cost", "Delivery_Zone"]
NUMERIC = FEATURES[:4]
TARGET = "Delivery_Time_Days"

def evaluate(label, actual, predicted):
    return {"Model": label,
            "MAE_days": mean_absolute_error(actual, predicted),
            "RMSE_days": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted)}

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    prep = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("zone", OneHotEncoder(handle_unknown="ignore"), ["Delivery_Zone"])
    ])
    lr = Pipeline([("preprocessor", prep), ("model", LinearRegression())])
    tree = Pipeline([("preprocessor", prep),
                     ("model", DecisionTreeRegressor(max_depth=4, random_state=42))])
    lr.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    rows = [
        evaluate("Linear Regression", y_test, lr.predict(X_test)),
        evaluate("Decision Tree", y_test, tree.predict(X_test))
    ]
    grid = GridSearchCV(
        tree,
        {"model__max_depth": [2, 3, 4, 5, None],
         "model__min_samples_leaf": [1, 2, 5, 10]},
        scoring="neg_mean_absolute_error", cv=5, n_jobs=1
    )
    grid.fit(X_train, y_train)
    rows.append(evaluate("Tuned Decision Tree", y_test,
                         grid.best_estimator_.predict(X_test)))
    results = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/model_comparison.csv", index=False)
    print(results.round(3).to_string(index=False))
    print("Best parameters:", grid.best_params_)
    print("Best CV MAE:", round(-grid.best_score_, 3), "days")
    print("Caution: simulated random delivery-time target; not an operational forecast.")

if __name__ == "__main__":
    main()
