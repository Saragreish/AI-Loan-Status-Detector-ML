# AI-Loan-Status-Detector-ML

This project focuses on predicting loan default using a decision tree classifier. It involves critical steps of data preprocessing and model building to determine the loan status.

### Project Goal
The primary goal of this project is to accurately check and predict loan status, specifically identifying potential loan defaults.

### Dataset
The dataset used in this project is sourced from `loan_data.csv`. It contains various features related to loan applicants and their loan details, including:
* `person_age`
* `person_gender`
* `person_education`
* `person_income`
* `person_emp_exp` (person employment experience)
* `person_home_ownership`
* `loan_amnt` (loan amount)
* `loan_intent`
* `loan_int_rate` (loan interest rate)
* `loan_percent_income`
* `cb_person_cred_hist_length` (credit history length)
* `credit_score`
* `previous_loan_defaults_on_file`
* `loan_status` (target variable)

### Processing
The data processing involves several key stages:

1.  **Data Loading and Preparation**:
    * The `loan_data.csv` file is loaded into a pandas DataFrame.
    * Column names are stripped of any leading/trailing whitespace.
    * Features (`X`) and the target variable (`Y`, which is `loan_status`) are separated.

2.  **Categorical Feature Encoding**:
    * `LabelEncoder` from `sklearn.preprocessing` is applied to transform categorical features into numerical representations. Based on the provided Python code (`loan.py`) and PDF, the following columns are encoded:
        * `person_gender` (at index 1 in `X`)
        * `person_education` (at index 2 in `X`)
        * `person_home_ownership` (at index 5 in `X`)
        * `loan_intent` (at index 7 in `X`)
        * `previous_loan_defaults_on_file` (at the last index `-1` in `X`)

3.  **Data Splitting**:
    * The dataset is split into training (`x_train`, `y_train`) and testing (`x_test`, `y_test`) sets using `train_test_split` from `sklearn.model_selection`.
    * A `test_size` of `0.2` (20% for testing) and `random_state` of `0` are used to ensure reproducibility.

4.  **Feature Scaling**:
    * `StandardScaler` from `sklearn.preprocessing` is used to scale the numerical features in both the training and testing sets. This step helps in bringing the values of different features to a similar scale, which can improve model performance.

### Model Building
A Decision Tree Classifier is used for building the prediction model:
* The model is initialized using `tree.DecisionTreeClassifier` from `sklearn.tree`.
* The `criterion` is set to `'entropy'`, and `random_state` is set to `0` for consistent results.
* The model is trained on the `x_train` and `y_train` datasets.
* Predictions are made on the `x_test` dataset.

### Get Accuracy
The model's performance is evaluated using `accuracy_score` from `sklearn.metrics`. The accuracy score represents the proportion of correctly predicted instances.

### How to Run
To run this project, ensure you have the necessary Python libraries installed.

1.  **Prerequisites**:
    * Python 3.x
    * `pandas`
    * `scikit-learn` (install via `pip install scikit-learn`)

2.  **Dataset**: Make sure `loan_data.csv` is in the same directory as `loan.py`.

3.  **Execution**: Run the `loan.py` script from your terminal:
    ```bash
    python loan.py
    ```
    The script will output the transformed `X` array during processing, and after the model is trained and predictions are made, the accuracy score is expected to be printed to the console (as indicated in the project's overview).

### Prepared By:
* Sara Ahmed 221006292
* Nagham Khaled 221006874
