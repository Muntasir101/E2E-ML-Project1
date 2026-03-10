## Guided ML Project 1

This repository contains **Guided Machine Learning Project 1**, developed in Python using a conda environment. The project code lives under the `src` package and is installable as a local package via `setup.py`.

### Project structure

- **`src/`**: Python source package for the project (add your modules and packages here).
- **`requirements.txt`**: Python dependencies used by the project.
- **`setup.py`**: Packaging script that installs the project and reads dependencies from `requirements.txt`.

### Environment setup (recommended: conda)

1. **Open Anaconda Prompt** and navigate to the project directory:

```bash
cd \Guided-Project1
```

2. **Create a conda environment** (run once):

```bash
conda create -n guided_ml_env python=3.12 -y
```

3. **Activate the environment** (every new session):

```bash
conda activate guided_ml_env
```

4. **Install dependencies and the project in editable mode**:

```bash
pip install -r requirements.txt
pip install -e .
```

After this, you can import your package in Python code or notebooks, for example:

```python
import src  # or from src import <your_module> once you add code
```

### Running code

From the activated environment in the project root:

- **Run a Python script** (once you create it under `src/` or the root):

```bash
python path\to\your_script.py
```

- **Start Jupyter Notebook (optional)**:

```bash
pip install notebook  # if not already installed
jupyter notebook
```

Then, in a notebook, ensure the kernel is using the `guided_ml_env` conda environment.

### Adding your ML code

1. Create modules inside `src/`, for example:
   - `src/data_preprocessing.py`
   - `src/model_training.py`
   - `src/evaluation.py`

2. Import them in scripts or notebooks, for example:

```python
from src.data_preprocessing import load_data
from src.model_training import train_model
```

3. Update `requirements.txt` whenever you add new dependencies, then rerun:

```bash
pip install -r requirements.txt
```

### Notes

- The `setup.py` uses `requirements.txt` for dependencies and installs the project as `guided_ml_project1`.
- The `-e .` entry in `requirements.txt` is handled automatically by `setup.py` and should be left as-is.

