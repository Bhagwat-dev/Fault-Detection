import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load datasets (replace with actual paths)
dataset2 = pd.read_csv(r"C:\Users\Bhagwat\Downloads\Dataset\classData.csv")  # Replace with actual path
dataset1 = pd.read_csv(r"C:\Users\Bhagwat\Downloads\Dataset\detect_dataset.csv")  # Replace with actual path

# Fault type mapping for Dataset 2
fault_mapping = {
    (0, 0, 0, 0): "No Fault",
    (1, 0, 0, 1): "LG Fault",
    (0, 0, 1, 1): "LL Fault",
    (1, 0, 1, 1): "LLG Fault",
    (0, 1, 1, 1): "LLL Fault",
    (1, 1, 1, 1): "LLLG Fault",
    (0, 1, 1, 0): "unmapped"
}

# Combine G, C, B, A columns into a tuple for fault mapping
binary_targets = dataset2[['G', 'C', 'B', 'A']]
dataset2['Fault_Type'] = binary_targets.apply(tuple, axis=1).map(fault_mapping)

# Check for unmapped fault types
if dataset2['Fault_Type'].isnull().any():
    unmapped_faults = binary_targets.apply(tuple, axis=1)[dataset2['Fault_Type'].isnull()]
    print("Unmapped fault types detected:\n", unmapped_faults.value_counts())
    raise ValueError("Unmapped fault types detected. Check fault_mapping and dataset.")

# Extract features and targets from Dataset 2
features_2 = dataset2[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]
targets_2 = dataset2['Fault_Type']

# Normalize features
scaler = StandardScaler()
features_2_scaled = scaler.fit_transform(features_2)

# Encode fault types as categorical labels
fault_labels = targets_2.astype('category').cat.codes
fault_label_mapping = dict(enumerate(targets_2.astype('category').cat.categories))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_2_scaled, fault_labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=fault_label_mapping.values()))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Visualize feature importance
importances = rf_model.feature_importances_
plt.bar(features_2.columns, importances)
plt.title("Feature Importance - Fault Classification")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Map predictions back to fault types
y_test_mapped = pd.Series(y_test).map(fault_label_mapping)
y_pred_mapped = pd.Series(y_pred).map(fault_label_mapping)

# Example: Show a comparison of actual vs predicted fault types
comparison = pd.DataFrame({'Actual': y_test_mapped, 'Predicted': y_pred_mapped})
print("\nSample Predictions:")
print(comparison.head())

# Detect faults using Dataset 1
features_1 = dataset1[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]
features_1_scaled = scaler.transform(features_1)  # Normalize features using the same scaler
predictions_1 = rf_model.predict(features_1_scaled)
predicted_fault_types = pd.Series(predictions_1).map(fault_label_mapping)

# Save the predictions to the original dataset1 for analysis
dataset1['Predicted_Fault_Type'] = predicted_fault_types
print("\nSample Fault Predictions on Detect Dataset:")
print(dataset1.head())
