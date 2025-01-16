# Fault Detection using Random Forest Classifier

This repository contains scripts and tools for detecting electrical faults using a Random Forest Classifier. The project focuses on analyzing fault data from electrical systems and predicting fault types based on currents and voltages. The model is trained using a dataset and evaluated for predictive accuracy.

---

## **Content**
- Power System was modeled in MATLAB to simulate fault analysis. The power system consists of 4 generators of 11 × 10^3 V, each pair located at each end of the transmission line. Transformers are present in between to simulate and study the various faults at the midpoint of the transmission line.
- The circuit was simulated under normal conditions as well as under various fault conditions. At the output side of the system voltage and current was measureed. 12000 data points were collected and lebelled.

---

## **Features**
1. **Fault Type Mapping:**
   - Maps binary columns (G, C, B, A) into fault types such as No Fault, LG Fault, LL Fault, etc.
   - Fault types are mapped based on combinations of binary features.

2. **Data Preprocessing:**
   - Standardization of current and voltage features (`Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`).
   - Fault type labeling for training and testing.

3. **Fault Prediction (Python):**
   - Utilizes the Random Forest Classifier to predict fault types based on input data.
   - Evaluates model performance using accuracy, confusion matrix, and classification report.

4. **Feature Importance Visualization:**
   - Displays a bar plot of feature importance, helping to identify which electrical features are most important for fault classification.

5. **Results Output:**
   - Displays model performance metrics such as accuracy, precision, recall, and F1-score.
   - Outputs predicted fault types for new datasets.

---

## **Requirements**
- Python (version >= 3.8) with the following libraries:
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib

---

## **Fault Type Examples**
The fault types are classified based on combinations of four binary features (`G`, `C`, `B`, `A`). Below are the fault types and their corresponding binary combinations:

- `[0 0 0 0]` - **No Fault**
- `[1 0 0 1]` - **LG Fault** (Between Phase A and Ground)
- `[0 0 1 1]` - **LL Fault** (Between Phase A and Phase B)
- `[1 0 1 1]` - **LLG Fault** (Between Phases A, B and Ground)
- `[0 1 1 1]` - **LLL Fault** (Between All Three Phases)
- `[1 1 1 1]` - **LLLG Fault** (Three Phase Symmetrical Fault)
- `[0 1 1 0]` - **Unmapped** (Others)

---

## **Results**
Key results include:
- Model evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- Feature importance bar plot indicating the significance of each electrical feature in fault classification.
- Sample predictions showing the comparison of actual vs. predicted fault types.
