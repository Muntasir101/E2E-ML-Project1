## Data Transformation Component (`src/components/data_transformation.py`)

### High-level purpose

The data transformation component takes the **raw, ingested training and testing CSVs** (e.g., `artifacts/train.csv` and `artifacts/test.csv`), **builds a preprocessing pipeline**, applies that pipeline to the input features, and **returns transformed NumPy arrays** along with a **serialized preprocessing object**.  

This prepares the data so that a model can consume **clean, numerically encoded, and scaled** features, while also ensuring that the exact same preprocessing can be reused later (for validation, testing, or serving).

---

## 1. Imports and why they are needed

```python
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object
```

- **`sys`**
  - Used when wrapping exceptions in `CustomException(e, sys)`.
  - Allows the custom exception class to access traceback and other interpreter-level details to produce rich error reports.

- **`dataclass`**
  - Used to define `DataTransformationConfig` as a simple configuration holder with minimal boilerplate.
  - Makes it clear that this class exists purely to store configuration values (in this case, the preprocessor file path).

- **`numpy as np`**
  - Used to manipulate arrays and to horizontally concatenate features and target:
    - `np.c_[input_feature_train_arr, np.array(target_feature_train_df)]`
  - This produces a single NumPy array where the last column is the target (`math_score`) and the preceding columns are transformed features.

- **`pandas as pd`**
  - Used to read the training and test CSV files into `DataFrame` objects (`train_df`, `test_df`).
  - Also used to manipulate columns (dropping the target column to form feature sets).

- **Scikit-learn components**
  - **`ColumnTransformer`**
    - Coordinates applying different transformation pipelines to different subsets of columns (numerical vs categorical).
    - Ensures that the preprocessing steps are applied consistently and in the right order to the correct columns.
  - **`SimpleImputer`**
    - Fills in missing values:
      - For numerical columns: uses the median.
      - For categorical columns: uses the most frequent category.
    - Prevents downstream steps (like scaling or encoding) from failing due to NaNs.
  - **`Pipeline`**
    - Chains multiple preprocessing steps (e.g., imputation → scaling) into a single object.
    - Guarantees that during both `fit_transform` (on training data) and `transform` (on test or new data), the same sequence of operations is applied.
  - **`OneHotEncoder`**
    - Converts categorical string columns into a numerical, one-hot encoded matrix (0/1 columns for each category).
    - Necessary because most models cannot directly handle raw strings.
  - **`StandardScaler`**
    - For numerical columns:
      - Centers and scales features to have zero mean and unit variance (improving model training stability).
    - For categorical (one-hot encoded) columns:
      - Used with `with_mean=False` so that sparse matrices are not densified.
      - Scales the binary indicator columns without subtracting a mean.

- **`CustomException` and `logging`**
  - Provide standardized error handling and logging across the project.
  - Logging records steps such as “Read train and test data completed” and “Applying preprocessing object...”, helping trace pipeline execution.

- **`os`**
  - Used to construct the file path for the saved preprocessor object under the `artifacts` directory.

- **`save_object` from `src.utils`**
  - A utility function for serializing and saving Python objects (likely via `pickle` or similar).
  - Used to persist the fitted preprocessing pipeline to disk so the same transformations can be reused later (e.g., at prediction time).

---

## 2. `DataTransformationConfig` dataclass

```python
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
```

- **Purpose**:
  - Holds configuration related to data transformation, currently just the path where the preprocessor object will be saved.

- **Field**:
  - `preprocessor_obj_file_path`:
    - Default: `artifacts/proprocessor.pkl`.
    - Meaning: When the preprocessing pipeline is fitted, it will be serialized and stored at this location.
    - Design implication: Downstream steps (e.g., model training or serving) can load this exact object and apply consistent preprocessing to new data.

- **Why a dataclass**:
  - Keeps configuration explicit and structured.
  - Makes it easy to extend in the future (e.g., adding different preprocessor versions or additional artifact paths).

---

## 3. `DataTransformation` class

```python
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
```

- **Role**:
  - Encapsulates all logic related to transforming raw tabular data into model-ready numeric arrays.

- **Constructor behavior**:
  - Instantiates `DataTransformationConfig` and stores it as `self.data_transformation_config`.
  - This provides an easy way for methods to access configuration like the preprocessor save path without hardcoding it repeatedly.

---

## 4. `get_data_transformer_object` method (building the preprocessing pipeline)

```python
def get_data_transformer_object(self):
    '''
    This function si responsible for data trnasformation
    '''
    try:
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())

            ]
        )

        cat_pipeline=Pipeline(

            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]

        )

        logging.info(f"Categorical columns: {categorical_columns}")
        logging.info(f"Numerical columns: {numerical_columns}")

        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)

            ]


        )

        return preprocessor
    
    except Exception as e:
        raise CustomException(e,sys)
```

### 4.1 Column definitions

- **`numerical_columns = ["writing_score", "reading_score"]`**
  - These are the raw numeric features that will be:
    - Imputed (missing values filled with median).
    - Scaled (standardized).

- **`categorical_columns = [...]`**
  - These are string or categorical features:
    - `gender`
    - `race_ethnicity`
    - `parental_level_of_education`
    - `lunch`
    - `test_preparation_course`
  - They cannot be fed directly to most ML algorithms; they require encoding as numeric features.

Defining these lists explicitly ensures the transformations are **column-name driven**, not position-based, which is safer and more maintainable when schemas evolve.

### 4.2 Numerical pipeline (`num_pipeline`)

```python
num_pipeline= Pipeline(
    steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ]
)
```

- **Step 1: `SimpleImputer(strategy="median")`**
  - Fills missing numeric values with the median of each column (computed from the training data).
  - Why median:
    - Robust to outliers (unlike mean).
    - Keeps the distribution more stable when there are extreme values.

- **Step 2: `StandardScaler()`**
  - Transforms each numerical feature to have (approximately) zero mean and unit variance based on the training data.
  - Why scaling:
    - Many models (e.g., gradient-based, distance-based) benefit from features being on similar scales.
    - Prevents features with large numeric ranges from dominating the learning process.

The `Pipeline` ensures that during training, the imputer is **fitted and then applied**, and during inference or on test data, the same imputation and scaling parameters learned from training data are reused.

### 4.3 Categorical pipeline (`cat_pipeline`)

```python
cat_pipeline=Pipeline(
    steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder",OneHotEncoder()),
    ("scaler",StandardScaler(with_mean=False))
    ]
)
```

- **Step 1: `SimpleImputer(strategy="most_frequent")`**
  - Replaces missing category values with the most frequent category observed in the training data for each column.
  - This is a reasonable default when missingness does not carry specific semantic meaning.

- **Step 2: `OneHotEncoder()`**
  - Converts each categorical column into multiple binary indicator columns (one per category).
  - Example: a `gender` column with values `["male", "female"]` may become two columns: `gender_male`, `gender_female`.
  - Ensures categorical information is represented numerically without imposing an arbitrary ordering.

- **Step 3: `StandardScaler(with_mean=False)`**
  - Scales each one-hot encoded column, but without subtracting the mean (`with_mean=False`) because:
    - One-hot encoded outputs are often stored in a sparse matrix.
    - Subtracting the mean would densify the matrix, greatly increasing memory usage.
  - This keeps the representation efficient and can still normalize the scale of features if desired.

### 4.4 Combining pipelines with `ColumnTransformer`

```python
preprocessor=ColumnTransformer(
    [
    ("num_pipeline",num_pipeline,numerical_columns),
    ("cat_pipelines",cat_pipeline,categorical_columns)
    ]
)
```

- **What `ColumnTransformer` does**:
  - Applies `num_pipeline` only to the `numerical_columns`.
  - Applies `cat_pipeline` only to the `categorical_columns`.
  - Concatenates the outputs into a single, unified feature matrix.

- **Logging**:
  - `logging.info(f"Categorical columns: {categorical_columns}")`
  - `logging.info(f"Numerical columns: {numerical_columns}")`
  - These logs help confirm exactly which columns are treated as numeric vs categorical at runtime.

- **Return value**:
  - The method returns the **unfitted** `preprocessor` object.
  - Fitting occurs later in `initiate_data_transformation`, ensuring training data is used to fit all preprocessing parameters.

---

## 5. `initiate_data_transformation` method

```python
def initiate_data_transformation(self,train_path,test_path):

    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)

        logging.info("Read train and test data completed")

        logging.info("Obtaining preprocessing object")

        preprocessing_obj=self.get_data_transformer_object()

        target_column_name="math_score"
        numerical_columns = ["writing_score", "reading_score"]

        input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
        target_feature_test_df=test_df[target_column_name]

        logging.info(
            f"Applying preprocessing object on training dataframe and testing dataframe."
        )

        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        logging.info(f"Saved preprocessing object.")

        save_object(

            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj

        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )
    except Exception as e:
        raise CustomException(e,sys)
```

### 5.1 Reading train and test CSVs

- `train_df = pd.read_csv(train_path)`  
- `test_df = pd.read_csv(test_path)`

- **Purpose**:
  - Load the ingested artifacts (`train.csv`, `test.csv`) back into memory as DataFrames.

- **Logging**:
  - `logging.info("Read train and test data completed")` confirms both reads succeeded.

### 5.2 Getting the preprocessing object

- `logging.info("Obtaining preprocessing object")`
- `preprocessing_obj = self.get_data_transformer_object()`

- **What happens**:
  - Calls the previously defined method to build the `ColumnTransformer` with numerical and categorical pipelines.
  - The returned `preprocessing_obj` is not yet fitted.

### 5.3 Defining target and feature structure

- `target_column_name = "math_score"`
  - This is the **label** the model will be trained to predict.

- `numerical_columns = ["writing_score", "reading_score"]`
  - Matches the earlier definition in `get_data_transformer_object` for consistency.

- **Splitting into input features and target**:
  - Training:
    - `input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)`
    - `target_feature_train_df = train_df[target_column_name]`
  - Testing:
    - `input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)`
    - `target_feature_test_df = test_df[target_column_name]`

- **Why this split**:
  - Models train on features (`X`) and predict targets (`y`).
  - Removing `math_score` from the input feature DataFrames ensures there is no target leakage.

### 5.4 Applying preprocessing

- Logging:
  - `logging.info("Applying preprocessing object on training dataframe and testing dataframe.")`

- Fitting and transforming:
  - `input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)`
    - On the **training** features:
      - `fit_transform`:
        - Fits imputers (median and most frequent), encoders, and scalers to the training data.
        - Applies the learned transformations to the same training data.
  - `input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)`
    - On the **test** features:
      - Only `transform` is called:
        - Uses the parameters learned from the training data (e.g., medians, category indices, scaling factors).
        - Guarantees that test data is processed using exactly the same transformation rules.

- **Why this distinction (`fit_transform` vs `transform`) matters**:
  - Fitting on test data would leak information from evaluation data into the model training process, biasing performance.
  - This design enforces that **all preprocessing parameters are derived solely from the training data**.

### 5.5 Combining features and targets into arrays

- Training:
  ```python
  train_arr = np.c_[
      input_feature_train_arr, np.array(target_feature_train_df)
  ]
  ```
- Testing:
  ```python
  test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
  ```

- **What `np.c_` does**:
  - Horizontally concatenates arrays.
  - Here, it appends the target vector as the **last column** of the transformed feature matrix.

- **Why combine them**:
  - Some downstream training utilities or custom training code may expect a single NumPy array where:
    - Columns `0..n-1` are features.
    - The last column is the target.
  - This convention can simplify saving, batching, or custom training loops.

### 5.6 Saving the fitted preprocessor

- Logging:
  - `logging.info(f"Saved preprocessing object.")`

- Saving:
  ```python
  save_object(
      file_path=self.data_transformation_config.preprocessor_obj_file_path,
      obj=preprocessing_obj
  )
  ```

- **What happens**:
  - The **fitted** `preprocessing_obj` (with all learned parameters) is serialized to `artifacts/proprocessor.pkl`.
  - This file can later be loaded when making predictions on new data to ensure that:
    - The new data is transformed with the exact same preprocessing rules used during training.

### 5.7 Return values

- The method returns a tuple:
  ```python
  (
      train_arr,
      test_arr,
      self.data_transformation_config.preprocessor_obj_file_path,
  )
  ```

- **Meaning**:
  - `train_arr`: Fully transformed training data with target appended as the last column.
  - `test_arr`: Fully transformed testing data with target appended.
  - `preprocessor_obj_file_path`: Path to the saved preprocessor object (`proprocessor.pkl`).

- **Why this is useful**:
  - The next pipeline component (e.g., model training) can:
    - Use `train_arr` and `test_arr` directly as numeric input.
    - Persist or load the preprocessor object when needed (e.g., at inference time).

### 5.8 Error handling

- The entire method is wrapped in a `try/except`:
  ```python
  except Exception as e:
      raise CustomException(e,sys)
  ```

- **Effect**:
  - Any failure (file read errors, shape mismatches, sklearn issues, etc.) is caught and re-raised as a `CustomException`.
  - This keeps error reporting consistent and allows higher-level orchestration code to handle all transformation-related errors through one exception type.

---

## Summary of data transformation flow

1. **Read** the ingested train and test CSVs into Pandas DataFrames.
2. **Build** a composite preprocessing pipeline:
   - Numerical: median imputation → standard scaling.
   - Categorical: most frequent imputation → one-hot encoding → scaling (without centering).
3. **Split** each DataFrame into input features and target (`math_score`).
4. **Fit** the preprocessor on training features and **transform** both training and test features.
5. **Concatenate** transformed features and target columns into `train_arr` and `test_arr`.
6. **Save** the fitted preprocessor object to `artifacts/proprocessor.pkl`.
7. **Return** the transformed arrays and the path to the preprocessor for use in subsequent stages (e.g., model training and inference).

