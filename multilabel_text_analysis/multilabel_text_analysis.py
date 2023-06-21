import argparse
import pickle
import string
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("maxent_ne_chunker")
nltk.download("wordnet")

parser = argparse.ArgumentParser(description="Processes the data.")

parser.add_argument(
    "database_filepath",
    action="store",
    metavar='["/path/to/database.db"]',
    help="Provide the location of the database",
)

parser.add_argument(
    "model_filepath",
    action="store",
    metavar='["/path/to/model"]',
    help="Provide the destination of the produced pickle file",
)

rm = set(stopwords.words("english"))


def load_data(database_filepath):
    """
    Loads a pandas DataFrame from a sqlite database
    Args:
    database_filepath: path of the sqlite database
    Returns:
    x: features (data frame)
    y: target categories (data frame)
    category_names: index list with names of categories
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("data", engine)
    x = df[["message", "len"]]
    y = df.drop(
        ["id", "message", "original", "len", "genre_news", "genre_social"], axis=1
    )
    category_names = y.columns
    return x, y, category_names


def tokenize(text):
    """
    Tokenizes input text
    Args:
    text: text data as str
    Returns:
    text: tokenized text
    """
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    text = word_tokenize(text)
    text = list(set(text) - rm)
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text


text_transformer = Pipeline(
    [("vecttext", CountVectorizer(tokenizer=tokenize)), ("tfidf", TfidfTransformer())]
)

preprocessor = ColumnTransformer(
    [("text", text_transformer, "message")], remainder="passthrough"
)


def build_model():
    """
    Creates a pipeline for model training including a GridSearchCV object.

    Returns:
    cv: GridSearchCV object
    """
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("clf", RandomForestClassifier())]
    )

    parameters = [
        {
            "clf": [RandomForestClassifier()],
            "clf__n_estimators": [5, 50, 100, 250],
            "clf__max_depth": [5, 8, 10],
            "preprocessor__text__vecttext__max_df": [0.5, 0.75, 1.0],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
            "clf__random_state": [42],
        },
        {
            "clf": [MultiOutputClassifier(LinearSVC())],
            "clf__estimator__C": [1.0, 10.0, 100.0, 1000.0],
            "clf__estimator__max_iter": [5000],
            "preprocessor__text__vecttext__max_df": [0.5, 0.75, 1.0],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
            "clf__estimator__random_state": [42],
        },
        {
            "clf": [MultiOutputClassifier(LogisticRegression())],
            "clf__estimator__penalty": ["l1", "l2"],
            "clf__estimator__C": [0.01, 0.1, 1, 10, 100],
            "preprocessor__text__vecttext__max_df": [0.5, 0.75, 1.0],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
            "clf__estimator__random_state": [42],
        },
        {
            "clf": [MultiOutputClassifier(MultinomialNB())],
            "preprocessor__text__vecttext__max_df": [0.5, 0.75, 1.0],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
        },
    ]

    rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)

    cv = GridSearchCV(
        pipeline,
        parameters,
        cv=rkf,
        scoring=["f1_weighted", "f1_micro", "f1_samples"],
        refit="f1_weighted",
        n_jobs=-1,
    )

    return cv


def evaluate_model(model, x, y):
    """
    Prints the results of the GridSearchCV function. Predicts
    a test set and prints a classification report. Saves output
    in ./models/cv_results.txt.

    Arguments:
    model: trained sci-kit learn estimator
    x: feature data frame for test set evaluation
    y: target data frame for test set evaluation
    """

    df = pd.DataFrame.from_dict(model.cv_results_)
    print("##### Cross-validation results on validation set #####")
    print("Best score:{}".format(model.best_score_))
    print("Best parameters set:{}".format(model.best_estimator_.get_params()["clf"]))
    print("mean_test_f1_weighted: {}".format(df["mean_test_f1_weighted"]))
    print("mean_test_f1_micro: {}".format(df["mean_test_f1_micro"]))
    print("mean_test_f1_micro: {}".format(df["mean_test_f1_micro"]))
    print("mean_test_f1_samples: {}\n".format(df["mean_test_f1_samples"]))
    print("##### Scoring on test set #####")
    preds = model.predict(x)
    print(
        "Test set classification report: {}".format(
            classification_report(y, preds, target_names=list(y.columns))
        )
    )

    with open("./models/cv_results.txt", "w") as w:
        w.write(
            "##### Cross-validation results on validation set #####\n"
            "\n"
            "Best score: {}\n"
            "Best parameter set: {}\n"
            "mean_test_f1_weighted: {}\n"
            "mean_test_f1_micro: {}\n"
            "mean_test_f1_samples:{}\n"
            "\n"
            "\n"
            "##### Scoring on test set #####\n"
            "\n"
            "Test set classification report: \n"
            "{}\n".format(
                model.best_score_,
                model.best_estimator_.get_params()["clf"],
                df["mean_test_f1_weighted"].values[0],
                df["mean_test_f1_micro"].values[0],
                df["mean_test_f1_samples"].values[0],
                str(classification_report(y, preds, target_names=list(y.columns))),
            )
        )
    print("##### Results stored in ./models/results.txt #####")


def save_model(model, model_filepath):
    """
    Saves model as a .pkl file. Destination is set by model_filepath argument.

    Arguments:
    model: trained sci-kit learn estimator to save
    model_filepath: destination for model save
    """

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    """
    Loads the data, splits it into a train (80%) and test set (20%),
    trains the model with a GridSearchCV pipeline,
    evaluates it on the test set, and saves the model as a .pkl file.
    """
    if len(sys.argv) == 3:
        args = parser.parse_args()

        database_filepath = args.database_filepath
        model_filepath = args.model_filepath

        print("Loading data...\n    DATABASE: {}".format(database_filepath))

        x, y, category_names = load_data(database_filepath)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(x_train, y_train)

        print("Evaluating model...")
        evaluate_model(model, x_test, y_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            """
        Please provide the filepath of the disaster messages database\n
        as the first argument and the filepath of the pickle file to\n
        save the model to as the second argument. \n\nExample: python\n
        train_classifier.py ../data/DisasterResponse.db classifier.pkl
        """
        )


if __name__ == "__main__":
    main()
