import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ 1. Load the dataset ------------------
df = pd.read_csv('employee_data.csv')  # Ensure this file exists in same folder

# ------------------ 2. Preprocessing ------------------
df = df.dropna()  # Drop missing values

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------ 3. Feature Selection ------------------
X = df.drop('Attrition', axis=1)  # Features
y = df['Attrition']               # Target (0=Stay, 1=Leave)

# ------------------ 4. Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ 5. Train Model ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------ 6. Evaluate Model ------------------
y_pred = model.predict(X_test)
print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ------------------ 7. Feature Importance Chart ------------------
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# ------------------ 8. Predict on Full Dataset ------------------
df['Predicted_Attrition'] = model.predict(X)
df['Prediction_Prob'] = model.predict_proba(X)[:, 1]  # Probability of leaving
df['Prediction_Result'] = df['Predicted_Attrition'].apply(lambda x: 'Will Leave' if x == 1 else 'Will Stay')

# ------------------ 9. Export Results to CSV ------------------
output_filename = "employee_predictions.csv"
df.to_csv(output_filename, index=False)
print(f"\n✅ Prediction results saved to: {output_filename}")

# ------------------ 10. Show Sample Predictions Cleanly ------------------
sample_output = df[['EmployeeNumber', 'Age', 'MonthlyIncome', 'OverTime', 'Prediction_Result', 'Prediction_Prob']].head(10)
sample_output['Prediction_Prob'] = sample_output['Prediction_Prob'].round(2)  # optional formatting

print("\n📥 Sample Prediction Results:\n")
print(sample_output.to_string(index=False))
