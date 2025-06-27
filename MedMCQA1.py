import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('validation.csv')

# Create a column to check correctness
df['Correct'] = df['Prediction'] == df['GroundTruth']

# Accuracy chart
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Correct')
plt.title('Correct vs Incorrect Predictions')
plt.xlabel('Prediction Correctness')
plt.ylabel('Count')
plt.show()

# Confidence distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Confidence'], bins=10, kde=True)
plt.title('Model Confidence Distribution')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.show()

# Efficiency line plot
plt.figure(figsize=(6,4))
plt.plot(df['id'], df['Time_Taken'], marker='o')
plt.title('Inference Time per Record')
plt.xlabel('Record ID')
plt.ylabel('Time Taken (s)')
plt.show()
