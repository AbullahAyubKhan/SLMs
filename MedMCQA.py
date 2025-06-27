import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('validation.csv')

# Assume your CSV has Prediction and GroundTruth columns
df['Correct'] = df['Prediction'] == df['GroundTruth']

# Count correct vs incorrect
summary = df['Correct'].value_counts().rename({True: 'Correct', False: 'Incorrect'})

# Bar chart
plt.figure(figsize=(6,4))
summary.plot(kind='bar', color=['green', 'red'])
plt.title('Correct vs Incorrect Predictions')
plt.xlabel('Prediction Accuracy')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
