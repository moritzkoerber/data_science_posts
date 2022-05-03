import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

titanic = pd.read_csv("./titanic.csv")

X = titanic.drop("survived", axis=1)
y = titanic.survived

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

categorical_features = ["pclass", "sex", "embarked"]
categorical_transformer = Pipeline(
    [
        ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

numeric_features = ["age", "sibsp", "parch", "fare"]
numeric_transformer = Pipeline(
    [("imputer_num", SimpleImputer()), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    [
        ("categoricals", categorical_transformer, categorical_features),
        ("numericals", numeric_transformer, numeric_features),
    ],
    remainder="drop",
)

pipeline = Pipeline([("preprocessing", preprocessor), ("clf", LogisticRegression())])

params = [
    {
        "clf": [LogisticRegression()],
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__random_state": [42],
        "preprocessing__numericals__scaler": [StandardScaler(), "passthrough"],
        "preprocessing__numericals__imputer_num__strategy": ["mean", "median"],
    },
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [5, 50, 100, 250],
        "clf__max_depth": [5, 8, 10],
        "clf__random_state": [42],
        "preprocessing__numericals__scaler": [StandardScaler(), "passthrough"],
        "preprocessing__numericals__imputer_num__strategy": ["mean", "median"],
    },
]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

cv = GridSearchCV(
    pipeline, params, cv=rskf, scoring=["f1", "accuracy"], refit="f1", n_jobs=-1
)

cv.fit(X_train, y_train)

print(f"Best F1-score: {cv.best_score_:.3f}\n")
print(f"Best parameter set: {cv.best_params_}\n")
print(f"Scores: {classification_report(y_train, cv.predict(X_train))}")

preds = cv.predict(X_holdout)
print(f"Scores: {classification_report(y_holdout, preds)}\n")
print(f"F1-score: {f1_score(y_holdout, preds):.3f}")
