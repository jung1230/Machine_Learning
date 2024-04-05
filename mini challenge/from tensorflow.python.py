import os
import pandas as pd
import matplotlib.pyplot as plt


labels_df = pd.read_csv('purdue-face-recognition-challenge-2024/train.csv')


class_counts = labels_df['Category'].value_counts()


plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Number of Samples per Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
