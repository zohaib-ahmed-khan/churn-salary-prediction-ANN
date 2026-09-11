# 💼 Customer Estimated Salary Regression using ANN

An Artificial Neural Network (ANN) regression application designed to estimate customer salaries based on financial and demographic profiles using the `Churn_Modelling` dataset. Built with Keras/TensorFlow and deployed via Streamlit.

---

## 📌 Problem Being Solved
Predicting continuous numerical target values (`EstimatedSalary`) using customer financial and demographic parameters. This project demonstrates end-to-end machine learning engineering pipelines, rigorous data leakage prevention, model optimization, and clean deployment.

## 📊 Dataset Used
* **Dataset Name**: `Churn_Modelling.csv`
* **Target Variable**: `EstimatedSalary` (Continuous numerical target ranging from ~$11 to ~$200,000).

## 🛠️ Features Used
* **Numerical Predictors**: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `Exited`
* **Categorical Predictors**: `Geography`, `Gender`
* **Dropped Identifiers**: `RowNumber`, `CustomerId`, `Surname`

## ⚙️ Preprocessing Steps
1. **Pipeline Order**: Split into Train (80%) and Test (20%) sets **before** preprocessing to eliminate data leakage.
2. **One-Hot Encoding**: Fitted `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` strictly on training categorical data (`Geography`, `Gender`). Saved as `encoder.pkl`.
3. **Feature Scaling**: Fitted `StandardScaler()` strictly on the transformed training input matrix. Saved as `scaler.pkl`.

## 🏗️ Model Architecture
* **Framework**: TensorFlow / Keras Sequential API
* **Input Layer**: Dimension matching transformed feature size (12 inputs).
* **Hidden Layers**:
  * Dense (128 neurons, ReLU activation) + Dropout (0.15)
  * Dense (64 neurons, ReLU activation) + Dropout (0.10)
  * Dense (32 neurons, ReLU activation)
  * Dense (16 neurons, ReLU activation)
* **Output Layer**: 1 neuron with **Linear activation** (Continuous salary estimation).
* **Optimizer**: Adam (`learning_rate=0.001`)
* **Loss Function**: Mean Squared Error (MSE)

## 📈 Regression Metrics and Results
Evaluated on the **untouched test dataset**:
* **Mean Absolute Error (MAE)**: ~$50,134.28
* **Root Mean Squared Error (RMSE)**: ~$57,961.23
* **R² Score**: ~ -0.02 (near 0)

---

## 💻 How to Run the Project Locally

### Prerequisites
* Python 3.8+ installed

### Step-by-Step Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/zohaib-ahmed-khan/churn-salary-prediction-ANN.git)
   cd churn-salary-prediction-ANN

