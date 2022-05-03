# Tune your preprocessing steps and algorithm selection like hyperparameters

Using a pipeline to preprocess your data offers some substantive [advantages](https://moritzkoerber.github.io/python/tutorial/2019/10/11/blogpost/). A pipeline guarantees that no information from the test set is used in preprocessing or training the model. Pipelines are often combined with cross-validation to find the best parameter combination of a machine learning algorithm. However, the implemented preprocessing steps, for example whether to scale the data, or the implemented machine learning algorithm can also be seen as a hyperparameter; not of a single model but of the whole training process. We can therefore tune them as such to further improve our model's performance. In this post, I will show you how to do it with sci-kit learn!

We start with the required packages:


```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
```

We are again working with the Titanic data set.


```python
titanic = pd.read_csv('./titanic.csv')

titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>boat</th>
      <th>body</th>
      <th>home.dest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>Mahon Miss. Bridget Delia</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330924</td>
      <td>7.8792</td>
      <td>NaN</td>
      <td>Q</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Clifford Mr. George Quincy</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>110465</td>
      <td>52.0000</td>
      <td>A14</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Stoughton MA</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Yasbeck Mr. Antoni</td>
      <td>male</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>2659</td>
      <td>14.4542</td>
      <td>NaN</td>
      <td>C</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>Tenglin Mr. Gunnar Isidor</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>350033</td>
      <td>7.7958</td>
      <td>NaN</td>
      <td>S</td>
      <td>13 15</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>Kelly Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
      <td>NaN</td>
      <td>70.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Since we we will use the test data (in cross-validation) to make model-relevant decisions, such as what preprocessing steps we should perform, we need fresh, yet unseen data to obtain a valid estimate of our final model's out-of-sample performance. This is the same reason why we perform cross-validation in the first place! Nested cross-validation is an option here, but I leave it to creating a final hold-out set here:


```python
X = titanic.drop('survived', axis = 1)
y = titanic.survived

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)
```

[Following the last post](https://moritzkoerber.github.io/python/tutorial/2019/10/11/blogpost/), we create a pipeline including a ColumnTransformer ('preprocessor') that imputes the missing values, creates dummy variables for the categorical features and scales the numeric features.


```python
categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(
    [
        ('imputer_cat', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)

numeric_features = ['age', 'sibsp', 'parch', 'fare']
numeric_transformer = Pipeline(
    [
        ('imputer_num', SimpleImputer()),
        ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    [
        ('categoricals', categorical_transformer, categorical_features),
        ('numericals', numeric_transformer, numeric_features)
    ],
    remainder = 'drop'
)
```

In the end, we include this preprocessor in our pipeline.


```python
pipeline = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('clf', LogisticRegression())
    ]
)
```

## Tuning the machine learning algorithm

The same way we provide a list of hyperparameters of a machine learning algorithm in a parameter grid to find the best parameter combination, we can also fill in the machine learning algorithm itself as a "hyperparameter". `('clf', LogisticRegression())`above is simply a placeholder where other machine learning algorithms can be filled in. In the grid below, I first try out a logistic regression and, second, a random forest classifier. Note that the parameters need to be a list of dictionaries because both models possess different parameter values to tune.


```python
params = [
    {
        'clf': [LogisticRegression()],
        'clf__solver': ['liblinear'],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__random_state': [42],
    },
    {
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': [5, 50, 100, 250],
        'clf__max_depth': [5, 8, 10],
        'clf__random_state': [42],
    }
]
```

## Tuning the preprocessing steps

Next, we take care of tuning the preprocessing steps. We add them as parameters in the parameter grid by inserting their names given in the pipeline above: The `StandardScaler()` to preprocess numericals can be addressed by `'preprocessing__numericals__scaler'`. `'preprocessing'` addresses the pipeline step, which is our ColumnTransformer, `'__numericals'` adresses the pipeline for numericals inside this ColumnTransformer, and `'__scaler'` addresses the StandardScaler in this particular pipeline. We could modify the StandardScaler here, for example by giving `'preprocessing__scaler__with_std': ['False']`, but we can also set whether standardizing is performed at all. By passing the list `[StandardScaler(), 'passthrough']` to the `'scaler'` step, we either use the `StandardScaler()` in this step or no transformer at all (with `'passthrough'`). By this, we can evaluate how our model performance changes if we do not standardize at all! The same is true for the imputer: We can try out whether the mean or median deliver better performance in this particular cross-validation process.

Below you find the complete parameter grid with all mentioned parameters included:


```python
params = [
    {
        'clf': [LogisticRegression()],
        'clf__solver': ['liblinear'],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__random_state': [42],
        'preprocessing__numericals__scaler': [StandardScaler(), 'passthrough'],
        'preprocessing__numericals__imputer_num__strategy': ['mean', 'median']
    },
    {
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': [5, 50, 100, 250],
        'clf__max_depth': [5, 8, 10],
        'clf__random_state': [42],
        'preprocessing__numericals__scaler': [StandardScaler(), 'passthrough'],
        'preprocessing__numericals__imputer_num__strategy': ['mean', 'median']
    }
]
```

One last thing: If you wish to modify the `StandardScaler()`, e. g. by setting `with_mean`, you would need to do this at the last point where you declare what to fill into the `'scaler'` step. Here, this would be `'preprocessing__numericals__scaler': [StandardScaler(with_mean = False), 'passthrough']`.

Let's see what preprocessing steps and machine learning algorithm performs best:


```python
rskf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 42)

cv = GridSearchCV(pipeline, params, cv = rskf, scoring = ['f1', 'accuracy'], refit = 'f1', n_jobs = -1)

cv.fit(X_train, y_train)

print(f'Best F1-score: {cv.best_score_:.3f}\n')
print(f'Best parameter set: {cv.best_params_}\n')
print(f'Scores: {classification_report(y_train, cv.predict(X_train))}')
```

    Best F1-score: 0.722

    Best parameter set: {'clf': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=8, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=50,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False), 'clf__max_depth': 8, 'clf__n_estimators': 50, 'clf__random_state': 42, 'preprocessing__numericals__imputer_num__strategy': 'median', 'preprocessing__numericals__scaler': StandardScaler(copy=True, with_mean=True, with_std=True)}

    Scores:               precision    recall  f1-score   support

               0       0.87      0.95      0.91       647
               1       0.91      0.77      0.83       400

        accuracy                           0.88      1047
       macro avg       0.89      0.86      0.87      1047
    weighted avg       0.88      0.88      0.88      1047



Our best estimator is a random forest with `max_depth = 8`, `n_estimators = 50`, imputation by median and standardized numericals.

How do we do on completely new, yet unseen data?


```python
preds = cv.predict(X_holdout)
print(f'Scores: {classification_report(y_holdout, preds)}\n')
print(f'F1-score: {f1_score(y_holdout, preds):.3f}')
```

    Scores:               precision    recall  f1-score   support

               0       0.83      0.88      0.86       162
               1       0.79      0.71      0.75       100

        accuracy                           0.82       262
       macro avg       0.81      0.80      0.80       262
    weighted avg       0.82      0.82      0.81       262


    F1-score: 0.747


There seems to be some room for improvement!

Find the complete code in one single file here:
