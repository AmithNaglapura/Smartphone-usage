import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv('Smartphone_Usage_Productivity_Dataset_50000.csv')

# Initial inspection
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())





New :  
import matplotlib.pyplot as plt
import seaborn as sns

# Numeric columns only for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Examine distributions
df[numeric_cols].hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig('distributions.png')
