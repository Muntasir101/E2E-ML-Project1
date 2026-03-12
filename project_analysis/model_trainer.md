## Model Trainer Component (`src/components/model_trainer.py`)

### High-level purpose

The model trainer component takes **preprocessed training and testing arrays** (produced by the data transformation step), **trains multiple regression models with hyperparameter search**, selects the **best-performing model** based on evaluation metrics, **saves that best model** to disk as an artifact, and returns the **final R² score** on the test set.

It is the stage that turns cleaned, transformed data into a persisted machine learning model that can be reused later.

---

## 1. Imports and why they are needed

```python
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
```

- **`os`**
  - Used to construct the file path for saving the final trained model (`artifacts/model.pkl`).

- **`sys`**
  - Passed into `CustomException(e, sys)` when errors occur, so the custom exception class can access traceback/context information.

- **Model libraries**
  - **`CatBoostRegressor`** (CatBoost)
  - **`RandomForestRegressor`**, **`GradientBoostingRegressor`**, **`AdaBoostRegressor`** (from `sklearn.ensemble`)
  - **`DecisionTreeRegressor`** (from `sklearn.tree`)
  - **`LinearRegression`** (from `sklearn.linear_model`)
  - **`XGBRegressor`** (from XGBoost)
  - **`KNeighborsRegressor`** is imported but **not currently used** in the models dictionary.

  These provide a diverse set of regression algorithms, from simple linear models to powerful ensemble methods and gradient boosting frameworks. Having multiple model families lets you compare different inductive biases and pick the one that best fits the data.

- **`r2_score`**
  - Used to compute the coefficient of determination \(R^2\) on the test set, which measures how well the model explains the variance in the target.

- **`CustomException` and `logging`**
  - Provide standardized error handling and logging across the project.
  - Logging records key events such as splitting data and finding the best model.

- **`save_object` and `evaluate_models` from `src.utils`**
  - `save_object`:
    - Utility function for serializing and saving Python objects (models) to disk (likely with `pickle` or similar).
  - `evaluate_models`:
    - Central function responsible for:
      - Performing training and hyperparameter search for each candidate model.
      - Returning evaluation metrics (e.g., R² scores) for each model so they can be compared.

---

## 2. `ModelTrainerConfig` dataclass

```python
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
```

- **Purpose**:
  - Holds configuration related to model training artifacts, currently just the path where the final chosen model will be saved.

- **Field**:
  - `trained_model_file_path`:
    - Default: `artifacts/model.pkl`.
    - Meaning: When the best model is selected, it will be serialized and stored at this location.
    - Design implication: Downstream components (e.g., prediction services or evaluation scripts) can load the **exact same trained model** from this path.

- **Why a dataclass**:
  - Clearly expresses that this is a configuration object, not a behavior-heavy class.
  - Makes it easy to extend later (e.g., adding paths for multiple model versions or metrics).

---

## 3. `ModelTrainer` class

```python
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
```

- **Role**:
  - Encapsulates all logic for training and evaluating models, as well as saving the best model.

- **Constructor behavior**:
  - Instantiates `ModelTrainerConfig` and stores it as `self.model_trainer_config`.
  - This gives methods access to configuration like `trained_model_file_path` without hard-coding it.

---

## 4. `initiate_model_trainer` method

```python
def initiate_model_trainer(self,train_array,test_array):
    try:
        logging.info("Split training and test input data")
        X_train,y_train,X_test,y_test=(
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
        )
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }
        params={
            "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest":{
                'n_estimators': [8,16,32,64,128,256]
            },
            "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Linear Regression":{},
            "XGBRegressor":{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "CatBoosting Regressor":{
                'depth': [6,8,10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor":{
                'learning_rate':[.1,.01,0.5,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
        }

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                         models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        if best_model_score<0.6:
            raise CustomException("No best model found")
        logging.info(f"Best found model on both training and testing dataset")

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        predicted=best_model.predict(X_test)

        r2_square = r2_score(y_test, predicted)
        return r2_square
    except Exception as e:
        raise CustomException(e,sys)
```

### 4.1 Splitting input arrays into features and target

- Input arguments:
  - `train_array`, `test_array` are expected to be NumPy arrays where:
    - All columns except the last are **features**.
    - The last column is the **target** (e.g., `math_score`).

- The line:
  ```python
  X_train,y_train,X_test,y_test=(
      train_array[:,:-1],
      train_array[:,-1],
      test_array[:,:-1],
      test_array[:,-1]
  )
  ```

  does the following:

  - `train_array[:,:-1]`: all rows, all columns except the last → training features (`X_train`).
  - `train_array[:,-1]`: all rows, last column → training target (`y_train`).
  - `test_array[:,:-1]`: all rows, all columns except the last → test features (`X_test`).
  - `test_array[:,-1]`: all rows, last column → test target (`y_test`).

- **Why this design**:
  - Keeps a consistent, simple convention: the target is always at the last column.
  - The upstream transformation step is responsible for ensuring arrays are in this format.

### 4.2 Defining candidate models

- `models = { ... }` defines a dictionary mapping **human-readable model names** to **unfitted model instances**:

  - `"Random Forest"` → `RandomForestRegressor()`
  - `"Decision Tree"` → `DecisionTreeRegressor()`
  - `"Gradient Boosting"` → `GradientBoostingRegressor()`
  - `"Linear Regression"` → `LinearRegression()`
  - `"XGBRegressor"` → `XGBRegressor()`
  - `"CatBoosting Regressor"` → `CatBoostRegressor(verbose=False)`
  - `"AdaBoost Regressor"` → `AdaBoostRegressor()`

- **Why multiple models**:
  - Different algorithms capture different kinds of relationships:
    - Linear vs. non-linear.
    - Shallow vs. deep trees.
    - Bagging vs. boosting ensembles.
  - Trying several and comparing performance is a practical way to find a strong baseline model.

### 4.3 Hyperparameter grids (`params`)

- `params = { ... }` specifies a **hyperparameter search space** for each model:

  - **Decision Tree**:
    - `'criterion'`: loss function choices (`'squared_error'`, `'friedman_mse'`, `'absolute_error'`, `'poisson'`).
  - **Random Forest**:
    - `'n_estimators'`: number of trees in the forest (`[8,16,32,64,128,256]`).
  - **Gradient Boosting**:
    - `'learning_rate'`: step size for boosting (`[0.1, 0.01, 0.05, 0.001]`).
    - `'subsample'`: fraction of samples used for fitting each base learner (`[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]`).
    - `'n_estimators'`: number of boosting stages (`[8,16,32,64,128,256]`).
  - **Linear Regression**:
    - Empty dict `{}`: no hyperparameters to tune in this simple baseline.
  - **XGBRegressor**:
    - `'learning_rate'`: `[..., 0.001]`.
    - `'n_estimators'`: `[8,16,32,64,128,256]`.
  - **CatBoosting Regressor**:
    - `'depth'`: tree depth choices (`[6, 8, 10]`).
    - `'learning_rate'`: `[0.01, 0.05, 0.1]`.
    - `'iterations'`: number of boosting iterations (`[30, 50, 100]`).
  - **AdaBoost Regressor**:
    - `'learning_rate'`: `[0.1, 0.01, 0.5, 0.001]`.
    - `'n_estimators'`: `[8,16,32,64,128,256]`.

- **Why define grids here**:
  - Keeps all model configuration and search space together.
  - `evaluate_models` can generically loop over `models` and `params` to run hyperparameter optimization for each model.

### 4.4 Evaluating models with `evaluate_models`

- Call:
  ```python
  model_report: dict = evaluate_models(
      X_train=X_train,
      y_train=y_train,
      X_test=X_test,
      y_test=y_test,
      models=models,
      param=params,
  )
  ```

- **Expected behavior of `evaluate_models`** (based on its usage):
  - For each model in `models`:
    - Perform hyperparameter search (likely GridSearchCV / RandomizedSearchCV) using the corresponding hyperparameter grid from `params`.
    - Fit the model on `X_train`, `y_train`.
    - Evaluate the best-found configuration on `X_test`, `y_test`.
  - Return a dictionary like:
    ```python
    {
      "Random Forest": <r2_score_for_best_rf>,
      "Decision Tree": <r2_score_for_best_tree>,
      ...
    }
    ```

- **Why centralize this logic**:
  - Keeps the `ModelTrainer` clean; it only needs to interpret the report and choose the best model.
  - You can reuse `evaluate_models` for other tasks or experiments.

### 4.5 Selecting the best model

- Best score:
  ```python
  best_model_score = max(sorted(model_report.values()))
  ```
  - Takes all scores from the `model_report` dictionary, sorts them (ascending), then takes the maximum.
  - Equivalent to `max(model_report.values())`, but with an extra sort step (not necessary but harmless).

- Best model name:
  ```python
  best_model_name = list(model_report.keys())[
      list(model_report.values()).index(best_model_score)
  ]
  ```
  - Finds the index of `best_model_score` within the values list.
  - Uses that index to extract the corresponding key (model name) from the keys list.

- Best model instance:
  ```python
  best_model = models[best_model_name]
  ```

- **Why this approach**:
  - Provides a straightforward way to pick the top-performing model based on the evaluation metric.
  - The model name is useful for logging and debugging (knowing which algorithm won).

### 4.6 Thresholding and logging

- Threshold check:
  ```python
  if best_model_score<0.6:
      raise CustomException("No best model found")
  ```
  - If even the best model’s performance (e.g., R²) is below `0.6`, the code raises a `CustomException`.
  - Conceptually, this says: “No model reached a satisfactory performance level; do not proceed as if training succeeded.”

- Logging:
  ```python
  logging.info(f"Best found model on both training and testing dataset")
  ```
  - Indicates that a satisfactory model was found and passed the threshold.

### 4.7 Saving the best model

- Call:
  ```python
  save_object(
      file_path=self.model_trainer_config.trained_model_file_path,
      obj=best_model
  )
  ```

- **What happens**:
  - The **best-performing fitted model** is serialized and saved to `artifacts/model.pkl`.
  - This allows:
    - Reuse of the model in other scripts without retraining.
    - Deployment of the model for inference in production-like settings.

### 4.8 Final evaluation on test data

- Predictions:
  ```python
  predicted = best_model.predict(X_test)
  ```
  - Uses the saved best model to generate predictions on the test feature matrix.

- R² score:
  ```python
  r2_square = r2_score(y_test, predicted)
  return r2_square
  ```
  - Computes \(R^2\), which indicates how much of the variance in the target `y_test` is explained by the model predictions.
  - Returns this scalar metric as the final result of `initiate_model_trainer`.

- **Why compute R² again**:
  - Ensures the caller of `initiate_model_trainer` receives a simple, interpretable performance metric.
  - Even if `evaluate_models` used a similar metric internally, this explicit final evaluation documents the chosen model’s performance clearly at the trainer level.

### 4.9 Error handling

- Entire method is wrapped in:
  ```python
  except Exception as e:
      raise CustomException(e,sys)
  ```

- **Effect**:
  - Any error during:
    - Data splitting,
    - Evaluation,
    - Thresholding,
    - Saving,
    - or prediction
  - is captured and re-raised as a `CustomException`, preserving a consistent error-handling strategy across the pipeline.

---

## End-to-end role in the pipeline

1. **Input**: Receives `train_array` and `test_array` from the data transformation step, where:
   - All but the last column are transformed features.
   - The last column is the numeric target to be predicted.
2. **Model search**: Trains several regression algorithms with hyperparameter tuning using `evaluate_models`.
3. **Model selection**: Chooses the best-performing model according to the evaluation report.
4. **Quality check**: Enforces a minimum performance threshold; otherwise, raises an exception.
5. **Artifact saving**: Saves the best model to `artifacts/model.pkl`.
6. **Reporting**: Returns the final R² score on the test set for reporting and analysis.

