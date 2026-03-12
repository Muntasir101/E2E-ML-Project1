## Data Ingestion Component (`src/components/data_ingestion.py`)

### Purpose

The data ingestion component loads the raw student performance dataset, creates a copy of the raw data under the project `artifacts` directory, splits the data into training and testing subsets, and saves those splits for downstream steps (data transformation and model training).

### Imports and Dependencies

- **`os`**: Build OS-independent file paths and create directories.
- **`sys`**: Passed into `CustomException` to capture traceback information.
- **`CustomException` (`src.exception`)**: Project-specific exception wrapper to add context and consistent error handling.
- **`logging` (`src.logger`)**: Central logging utility used to track the progress and status of ingestion.
- **`pandas as pd`**: Used to load and write CSV files, and to represent tabular data as a `DataFrame`.
- **`train_test_split` (from `sklearn.model_selection`)**: Splits the dataset into training and testing subsets.
- **`dataclass`**: Used to define a simple, configuration-focused class for file paths.

### `DataIngestionConfig` Dataclass

```python
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
```

- **Role**: Centralizes configuration related to where the ingested data will be stored.
- **Fields**:
  - `train_data_path`: Output path for the training data CSV.
  - `test_data_path`: Output path for the testing data CSV.
  - `raw_data_path`: Output path for a full copy of the raw dataset.
- **Why a dataclass**: Reduces boilerplate, keeps configuration grouped in a single, easily extendable structure.

### `DataIngestion` Class

```python
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
```

- **Purpose**: Encapsulates all ingestion logic in a reusable class.
- **Constructor**: Instantiates `DataIngestionConfig` and stores it as `self.ingestion_config` so all methods can access the configured paths.

### `initiate_data_ingestion` Method

```python
def initiate_data_ingestion(self):
    logging.info("Entered the data ingestion method or component")
    try:
        df = pd.read_csv('notebook\\data\\stud.csv')
        logging.info('Read the dataset as dataframe')

        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

        logging.info("Train test split initiated")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        logging.info("Ingestion of the data is completed")

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
    except Exception as e:
        raise CustomException(e, sys)
```

Step-by-step behavior:

1. **Log entry into ingestion**  
   - Logs that ingestion has started, useful for tracing pipeline execution.

2. **Load raw dataset**  
   - `df = pd.read_csv('notebook\\data\\stud.csv')`  
   - Reads the original student performance data from `notebook\data\stud.csv` into a `DataFrame`.  
   - Logs successful loading of the dataset.

3. **Ensure output directory exists**  
   - `os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)`  
   - Creates the `artifacts` directory if it does not already exist, preventing file-write errors later.

4. **Persist a raw data copy**  
   - `df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)`  
   - Saves a full copy of the original dataset to `artifacts/data.csv`.  
   - This preserves the exact input used by the pipeline for reproducibility and auditing.

5. **Train/test split**  
   - Logs that train/test splitting is starting.  
   - `train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)`  
   - Splits the data into:
     - Training set: 80% of rows.
     - Test set: 20% of rows.  
   - `random_state=42` ensures deterministic and repeatable splits.

6. **Save training and test datasets**  
   - `train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)`  
   - `test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)`  
   - Writes the train and test datasets to `artifacts/train.csv` and `artifacts/test.csv`.  
   - Provides clean CSVs that downstream components can consume.

7. **Log completion and return paths**  
   - Logs that ingestion has completed successfully.  
   - Returns a tuple of `(train_data_path, test_data_path)` so the next pipeline step can use these file paths directly.

8. **Error handling**  
   - Wraps any exception raised within the method in `CustomException(e, sys)`, providing richer context and consistent error reporting across the project.

### Script Entry Point

```python
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
```

- Allows the module to be executed directly (e.g., `python -m src.components.data_ingestion`).
- When run as a script, it:
  - Instantiates `DataIngestion`.
  - Immediately triggers `initiate_data_ingestion`, performing the full ingestion workflow and writing artifacts.

### Outputs Produced

Running this component generates three key CSV files under the `artifacts` directory:

- **`data.csv`**: Full copy of the raw dataset as ingested.
- **`train.csv`**: Training subset (80% of the data).
- **`test.csv`**: Testing subset (20% of the data).

These outputs form the starting point for subsequent stages in the ML pipeline (feature engineering, transformation, model training, and evaluation).

---

## Deep Analysis of `data_ingestion.py`

### 1. Imports (what and why)

- **`os`**
  - Used for `os.path.join` to build OS-independent paths and `os.makedirs` to create directories.
  - Prevents path separator issues across Windows/Linux and avoids file-write errors when the `artifacts` directory does not exist.

- **`sys`**
  - Passed into `CustomException` so the custom exception class can access traceback information and provide rich error context.

- **`CustomException`**
  - A project-level wrapper for all ingestion errors, ensuring consistent formatting and handling of failures across the codebase.

- **`logging`**
  - Central logging utility that records key milestones: entering ingestion, successful CSV read, start of splitting, and completion.
  - Acts as a “timeline” of the ingestion step in your logs.

- **`pandas as pd`**
  - Handles reading and writing CSVs and holds tabular data as a `DataFrame`, which is the primary in-memory representation of the dataset.

- **`train_test_split`**
  - Provides a standard, well-tested way to split data into training and testing subsets.
  - Using `test_size=0.2` and `random_state=42` ensures a reproducible 80/20 partition.

- **`dataclass`**
  - Used to declare `DataIngestionConfig` in a concise, declarative style, automatically generating boilerplate like `__init__`.

### 2. `DataIngestionConfig` (configuration object)

- Holds three key paths:
  - `train_data_path`: where the training subset will be saved.
  - `test_data_path`: where the test subset will be saved.
  - `raw_data_path`: where a stable copy of the full raw dataset will be saved.
- Centralizing these values in a dataclass:
  - Avoids scattering hard-coded paths throughout the code.
  - Makes it easy to override paths for experiments or different environments.

### 3. `DataIngestion` class (encapsulation)

- The constructor creates a `DataIngestionConfig` instance and stores it in `self.ingestion_config`.
- This design:
  - Makes ingestion a reusable, stateful component that can be plugged into larger workflows.
  - Cleanly separates configuration (paths) from behavior (how data is read, split, and written).

### 4. `initiate_data_ingestion` (step-by-step logic)

1. **Log entry**
   - `logging.info("Entered the data ingestion method or component")`  
   - Marks the start of ingestion in logs so you can trace when this stage was executed.

2. **Load raw dataset**
   - `df = pd.read_csv('notebook\\data\\stud.csv')`  
   - Reads the CSV into a `DataFrame`. Any issues with file existence or format surface at this line.
   - `logging.info('Read the dataset as dataframe')` confirms that reading succeeded.

3. **Ensure output directory exists**
   - `os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)`  
   - Ensures the `artifacts` directory is present before writing any files, making the step idempotent and robust.

4. **Persist a raw data copy**
   - `df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)`  
   - Writes a complete copy of the original data to `artifacts/data.csv`, providing a fixed snapshot for reproducibility and auditing.

5. **Train/test split**
   - `logging.info("Train test split initiated")` logs the start of the split phase.
   - `train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)`  
   - Randomly partitions the dataset into 80% training and 20% testing, with a fixed seed to make the split repeatable.

6. **Save training and test subsets**
   - `train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)`  
   - `test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)`  
   - Writes `train.csv` and `test.csv` into `artifacts`, giving downstream steps standardized input files.

7. **Log completion and return paths**
   - `logging.info("Ingestion of the data is completed")` indicates success of the entire ingestion workflow.
   - Returning `(train_data_path, test_data_path)` makes it easy for the next pipeline stage to locate these artifacts without re-deriving paths.

8. **Error handling**
   - The `try/except` wraps all ingestion logic; any thrown exception is caught and re-raised as `CustomException(e, sys)`.
   - This centralizes error handling and ensures that all ingestion failures carry consistent, detailed context.

### 5. Script entry point

- The `if __name__ == "__main__":` block:
  - Instantiates `DataIngestion`.
  - Immediately calls `initiate_data_ingestion()`.
- This allows the file to act both as:
  - A standalone script for quickly generating artifacts from the raw CSV.
  - An importable module whose class and methods can be orchestrated in a larger training pipeline.

