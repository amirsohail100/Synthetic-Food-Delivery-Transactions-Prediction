# Synthetic Food Delivery Transactions Prediction

An end-to-end Machine Learning project analyzing synthetic food delivery transaction data. Evaluated custom ANN architectures vs. traditional regression models (Gradient Boosting, AdaBoost, Random Forest). Achieved a peak $R^2$ accuracy of **~0.678** through systematic tuning. Features saved artifacts (`model.pkl`, `scaler.pkl`, `columns.pkl`) for deployment.

---

## 🌟 Project Highlights

- **Comprehensive Evaluation**: Compared custom Artificial Neural Network (ANN) architectures against robust tree-based regression algorithms (Gradient Boosting, AdaBoost, Random Forest).
- **Optimization & Tuning**: Hyperparameter tuning applied to maximize predictive accuracy ($R^2  pprox 0.678$).
- **Deployment Ready**: Fully exported pipeline artifacts (`app.py`, `model.pkl`, `scaler.pkl`, `columns.pkl`) for real-time web inference.

---

## 📁 Repository Structure

```
Synthetic-Food-Delivery-Transactions-Prediction/
├── synthetic_food_delivery_transactions.csv  # Synthetic dataset used for training
├── model.ipynb                                # Jupyter Notebook with EDA, training & evaluation
├── app.py                                     # Streamlit / Web application for model deployment
├── model.pkl                                  # Best performing trained regression model
├── scaler.pkl                                 # Scaler artifact for feature normalization
├── columns.pkl                                # Saved feature list for inference alignment
├── requirements.txt                           # Project dependencies
└── README.md                                  # Documentation
```

---

## 🛠️ Models & Performance Summary

| Model / Architecture            | Peak Performance ($R^2$) |
| :------------------------------ | :----------------------: |
| **Tuned Ensemble Models / ANN** |        **~0.678**        |

_Evaluated algorithms include Gradient Boosting Regressor, AdaBoost Regressor, Random Forest Regressor, and custom ANN structures._

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/amirsohail100/Synthetic-Food-Delivery-Transactions-Prediction.git
cd Synthetic-Food-Delivery-Transactions-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web App (`app.py`)

```bash
streamlit run app.py
```

---

## 📊 Pipeline Workflow

1. **Data Preprocessing**: Handling missing values, scaling numerical features via `scaler.pkl`.
2. **Feature Alignment**: Using `columns.pkl` to maintain exact input column sequence during inference.
3. **Model Inference**: Loading `model.pkl` to predict transaction values/ranks instantly via `app.py`.

An end-to-end Machine Learning project analyzing synthetic food delivery transaction data. Evaluated custom ANN architectures vs traditional regression models (Gradient Boosting, AdaBoost, Random Forest). Achieved peak $R^2$ accuracy of ~0.678 through systematic tuning. Features saved artifacts (model.pkl, scaler.pkl, column.pkl) for deployment.

```

```
