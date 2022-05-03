import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

titanic = pd.read_csv("./titanic.csv")

categorical_features = ["pclass", "sex", "embarked"]
categorical_transformer = Pipeline(
    [
        ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

numeric_features = ["age", "sibsp", "parch", "fare"]
numeric_transformer = Pipeline(
    [("imputer_num", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    [
        ("categoricals", categorical_transformer, categorical_features),
        ("numericals", numeric_transformer, numeric_features),
    ],
    remainder="drop",
)

pipeline = Pipeline([("preprocessing", preprocessor), ("clf", LogisticRegression())])

params = {
    "clf__solver": ["liblinear"],
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__random_state": [42],
}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

cv = GridSearchCV(
    pipeline, params, cv=rskf, scoring=["f1", "accuracy"], refit="f1", n_jobs=-1
)

X = titanic.drop("survived", axis=1)
y = titanic.survived

cv.fit(X, y)

print(f"Best F1-score: {cv.best_score_:.3f}\n")
print(f"Best parameter set: {cv.best_params_}\n")
print(f"Scores: {classification_report(y, cv.predict(X))}")
