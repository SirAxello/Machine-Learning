# Malware Detection using Random Forest

This program demonstrates the use of the Random Forest algorithm to classify malware in a dataset. It includes data preprocessing, feature scaling, model training, evaluation, and visualization.

---

## Steps in the Program:

### 1. Import Libraries
The necessary Python libraries are imported:
- **pandas**: For data manipulation and analysis.
- **matplotlib** and **seaborn**: For data visualization.
- **scikit-learn**: For machine learning, preprocessing, and evaluation.

### 2. Load Dataset
The dataset is loaded using `pandas.read_csv()`. Ensure the dataset file path is correctly specified.

### 3. Data Preprocessing
- Handle missing values by replacing them with the mean of the respective columns.
- Encode the target column (`legitimate`) where:
  - `malware` = 1
  - `not malware` = 0
- Separate features (`X`) and target (`y`).
- Normalize numerical features using `StandardScaler`.

### 4. Train-Test Split
The dataset is split into training (70%) and testing (30%) sets using `train_test_split()`.

### 5. Random Forest Model
- A Random Forest classifier is created using `RandomForestClassifier` from scikit-learn.
- The model is trained using the training set.

### 6. Predictions
The trained Random Forest model predicts labels for the test set.

### 7. Model Evaluation
Key metrics for evaluating the model:
- **Accuracy**: Displayed in 4 decimal places.
- **Classification Report**: Shows precision, recall, F1-score, and support.
- **Confusion Matrix**: Displays the counts of true positive, true negative, false positive, and false negative predictions.

### 8. Visualizations
1. **Feature Distributions**: Histograms of feature values to understand data distribution.
2. **Confusion Matrix**: Heatmap for better visualization of model performance.

---

## How to Run the Program
1. Install required Python libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
