# COLING-Task-1
Here's a combined `README.md` file for all three notebooks, with sections that cover the overall setup, usage instructions, and details for each task.

---

# Machine Learning Tasks - Model Training and Evaluation

This project contains multiple Jupyter notebooks for training and evaluating machine learning models on different datasets, including English and multilingual data. The tasks include implementing K-Fold Cross-Validation, training various machine learning models, and handling multilingual data for language-specific model evaluation.

## Requirements

- **Python 3.x**
- **Jupyter Notebook**
- Libraries used (please install as needed):
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `transformers` (for multilingual models)
  - `numpy`
  
To install any missing packages, you can run:
```bash
pip install -r requirements.txt
```


## Files

1. **Notebooks**
   - `Task_1_Kfold_en.ipynb`: Implements K-Fold Cross-Validation on an English dataset.
   - `Task_1_ML_Models_en.ipynb`: Trains and evaluates multiple machine learning models on an English dataset.
   - `Task_1b_Multilingual.ipynb`: Trains models on multilingual datasets and evaluates performance per language.
   
2. **Data Files**
   - `*.json`: JSON files containing datasets and configurations for each task.

## Tasks Overview

### 1. Task 1 - K-Fold Cross-Validation (English Dataset)

This notebook performs K-Fold Cross-Validation to evaluate a model's performance on an English dataset. It splits the dataset into K folds, trains the model on each fold, and evaluates it to assess performance consistency.

- **Main Steps**:
  1. Load data from JSON files.
  2. Preprocess the data for model compatibility.
  3. Perform K-Fold Cross-Validation, training and evaluating the model across different folds.
  4. Aggregate and analyze results to obtain overall accuracy and consistency.

### 2. Task 1 - Machine Learning Models (English Dataset)

This notebook is dedicated to training and evaluating various machine learning models (e.g., logistic regression, XGBoost) on an English dataset. The goal is to identify the model that performs best based on selected metrics.

- **Main Steps**:
  1. Load and preprocess the dataset.
  2. Train multiple machine learning models.
  3. Evaluate each model on performance metrics (e.g., accuracy, precision).
  4. Analyze and compare results to select the optimal model.

### 3. Task 1b - Multilingual Model Training

This notebook trains machine learning models on multilingual datasets, aiming to achieve balanced accuracy across different languages. The dataset contains multiple language-specific entries, and the model is tested to ensure good performance for each language.

- **Main Steps**:
  1. Load and preprocess multilingual data from JSON files.
  2. Train a multilingual model or language-adaptive model.
  3. Evaluate model performance per language to check accuracy consistency.
  4. Analyze results across languages to improve language-specific performance.

## Usage Instructions

1. Ensure all required JSON files are in the same directory as the notebooks.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the desired notebook:
   - `Task_1_Kfold_en.ipynb` for K-Fold Cross-Validation
   - `Task_1_ML_Models_en.ipynb` for model training and comparison
   - `Task_1b_Multilingual.ipynb` for multilingual model evaluation
4. Run each cell sequentially to execute the entire process from data loading to result analysis.

## Notes

- **Parameter Tuning**: Modify hyperparameters in each notebook's model section as needed to improve performance.
- **Evaluation Metrics**: Results include metrics such as accuracy, precision, and F1-score for comprehensive performance analysis.
- **Cross-Language Consistency**: For multilingual models, pay close attention to performance consistency across languages, as it can vary significantly.

---

This `README.md` covers the purpose and use of each notebook while offering guidance on setup and execution. Let me know if you need further details or specific examples added.
