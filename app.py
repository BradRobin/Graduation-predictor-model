import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model_utils import save_model  # Import save_model function

# Load datasets
df_exams = pd.read_csv('exams.csv')
df_philippine_enrollment = pd.read_csv('Philippine Public School Student Enrollment and Teacher Employment Dataset.csv')
df_student_performance = pd.read_csv('Student_Performance.csv')

# Data Preprocessing
# Check data types to identify potential issues
print(df_exams.dtypes)  # Check data types in the exams dataset
print(df_student_performance.dtypes)  # Check data types in the performance dataset

# Convert numeric columns to proper numeric types if necessary
df_exams['math score'] = pd.to_numeric(df_exams['math score'], errors='coerce')
df_exams['reading score'] = pd.to_numeric(df_exams['reading score'], errors='coerce')
df_exams['writing score'] = pd.to_numeric(df_exams['writing score'], errors='coerce')

# Handle missing data (optional)
df_exams['math score'] = df_exams['math score'].fillna(df_exams['math score'].mean())
df_exams['reading score'] = df_exams['reading score'].fillna(df_exams['reading score'].mean())
df_exams['writing score'] = df_exams['writing score'].fillna(df_exams['writing score'].mean())

# Data Visualization
# Visualize the distribution of scores
plt.figure(figsize=(10,6))
sns.histplot(df_exams['math score'], kde=True, color='blue', label='Math Score')
sns.histplot(df_exams['reading score'], kde=True, color='red', label='Reading Score')
sns.histplot(df_exams['writing score'], kde=True, color='green', label='Writing Score')
plt.legend()
plt.title('Distribution of Scores')
plt.show()

# Data Processing for Student Performance (df_student_performance)
# Check if there are any categorical columns that need encoding
print(df_student_performance.dtypes)

# Convert categorical columns to numeric if needed (e.g., "Extracurricular Activities")
df_student_performance['Extracurricular Activities'] = df_student_performance['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Check if there are any missing values
print(df_student_performance.isnull().sum())

# Fill missing values or drop them
df_student_performance.fillna(df_student_performance.mean(), inplace=True)

# Split the dataset for training a model
X = df_student_performance[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = df_student_performance['Performance Index']  # Assuming 'Performance Index' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Logistic Regression (You can try other models as well)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted Low', 'Predicted High'], yticklabels=['Actual Low', 'Actual High'])
plt.title('Confusion Matrix')
plt.show()

# Visualize the performance predictions vs actual values
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, color='purple')
plt.title('Predicted vs Actual Performance Index')
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.show()

# Save the model
save_model(model)  # Use save_model function from model_utils
