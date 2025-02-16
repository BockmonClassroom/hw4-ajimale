import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""
Name: Abdikarim Jimale
Date: 02/12/2025
"""

# Load the Iris dataset
df = pd.read_csv("Iris.csv")

# Drop the 'Species' column for normalization and standardization
df_features = df.drop(columns=['Species'])

# Normalization 
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df_features), columns=df_features.columns)
# Round normalized value to one decimal place
df_normalized = df_normalized.round(1)
df_normalized['Species'] = df['Species']

# Standardization 
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df_features), columns=df_features.columns)
# Round standardized value to one decimal place
df_standardized = df_standardized.round(1)
df_standardized['Species'] = df['Species']

# Save the Normalized and standardized datasets to new csv files
df_normalized.to_csv('Iris_Normalized.csv', index=False)
df_standardized.to_csv('Iris_Standardized.csv', index=False)