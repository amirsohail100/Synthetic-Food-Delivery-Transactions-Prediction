# Synthetic Food Delivery Transactions Prediction

An end-to-end Machine Learning project analyzing synthetic food delivery transaction data. Evaluated custom ANN architectures vs. traditional regression models (Gradient Boosting, AdaBoost, Random Forest). Achieved a peak $R^2$ accuracy of **~0.678** through systematic tuning. Features saved artifacts (`model.pkl`, `scaler.pkl`, `columns.pkl`) for deployment.

---

## 💡 Key Lessons Learned & Challenges Overcome

During the initial phase of this project, a major hurdle was encountered regarding the dataset nature:

- **The Synthetic Data Surprise**: The dataset was directly picked up from Kaggle without initial deep correlation checks.
- **Correlation Insight**: When running feature correlation analysis (`df.corr()`), it became evident that the features had non-standard/synthetic relationships rather than real-world patterns, making traditional feature-target mappings tricky.
- **Project Pause & Pivot**: Due to the unexpected behavior of the synthetic dataset, the project was initially placed on hold.
- **Re-opening & Solution**: Later, the project was reopened with a fresh methodology. By systematically tuning tree-based ensemble models (Gradient Boosting, AdaBoost, Random Forest) and designing custom Artificial Neural Network (ANN) architectures, a peak $R^2$ accuracy of **~0.678** was successfully extracted from the complex dataset.

---

## 🌟 Project Highlights

- **Comprehensive Evaluation**: Compared custom Artificial Neural Network (ANN) architectures against robust tree-based regression algorithms (Gradient Boosting, AdaBoost, Random Forest).
- **Optimization & Tuning**: Systematic hyperparameter tuning applied to overcome synthetic data noise and maximize predictive power ($R^2 \approx 0.678$).
- **Deployment Ready**: Fully exported pipeline artifacts (`app.py`, `model.pkl`, `scaler.pkl`, `columns.pkl`) for real-time web inference.

---

## 📁 Repository Structure

```text
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
git clone [https://github.com/amirsohail100/Synthetic-Food-Delivery-Transactions-Prediction.git](https://github.com/amirsohail100/Synthetic-Food-Delivery-Transactions-Prediction.git)
cd Synthetic-Food-Delivery-Transactions-Prediction

An end-to-end Machine Learning project analyzing synthetic food delivery transaction data. Evaluated custom ANN architectures vs traditional regression models (Gradient Boosting, AdaBoost, Random Forest). Achieved peak $R^2$ accuracy of ~0.678 through systematic tuning. Features saved artifacts (model.pkl, scaler.pkl, column.pkl) for deployment.

```

```

```
