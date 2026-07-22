# Synthetic Food Delivery Transactions Rank Prediction

An advanced Multi-Input Deep Learning project built using TensorFlow/Keras that combines text processing and numerical features to predict delivery transaction rankings.

---

## 🌟 Overview

This project processes synthetic food delivery transaction data using a multi-branch neural network architecture. By merging textual sequence features with numerical transactional attributes, the model provides accurate rank predictions to optimize food delivery operations and ranking algorithms.

---

## 🏗️ Model Architecture

The model utilizes a dual-branch neural network design to handle multi-modal tabular and text data:

### 1. Text Feature Branch (`Text_Branch`)

- **Embedding Layer**: Converts text input sequences into dense vector representations.
- **Stacked Bidirectional GRU Layers**:
  - `Bidirectional(GRU(64, return_sequences=True))`
  - `Dropout(0.3)`
  - `Bidirectional(GRU(64, return_sequences=False))`
  - `Dropout(0.2)`
- **Dense Reduction Layers**: FC layers (`32`, `16`, `8` units with `ReLU`) to extract textual embeddings.

### 2. Numerical Feature Branch (`Numerical_Branch`)

- **Input**: 4 numerical features (`rating`, `variance`, `vote_count`, `year`).
- **Dense Layers with Batch Normalization & Dropout**:
  - `Dense(64)` -> `BatchNormalization` -> `Dropout(0.3)`
  - `Dense(32)` -> `BatchNormalization` -> `Dropout(0.2)`
  - `Dense(16)` -> `BatchNormalization`
  - `Dense(8)`

### 3. Merged Output & Head

- **Concatenation Layer**: Combines the 8-dimensional output vectors from both sub-models.
- **Final Regression Head**:
  - `Dense(16, activation="relu")`
  - `Dropout(0.1)`
  - `Dense(1, activation="linear", name="Rank_Output")`

```python
# Multi-Input Keras Architecture
merged_outputs = Concatenate()([text_out, num_out])
x = Dense(16, activation="relu")(merged_outputs)
x = Dropout(0.1)(x)
final_output = Dense(1, activation="linear", name="Rank_Output")(x)


An end-to-end Machine Learning project analyzing synthetic food delivery transaction data. Evaluated custom ANN architectures vs traditional regression models (Gradient Boosting, AdaBoost, Random Forest). Achieved peak $R^2$ accuracy of ~0.678 through systematic tuning. Features saved artifacts (model.pkl, scaler.pkl, column.pkl) for deployment.
```
