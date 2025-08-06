# phase2_eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("train.csv")

# Basic info
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Convert 'Date of Joining' to datetime and derive 'Tenure'
df['Date of Joining'] = pd.to_datetime(df['Date of Joining'], errors='coerce')
df['Tenure'] = pd.to_datetime('today').year - df['Date of Joining'].dt.year

# Drop 'Employee ID' (not useful for modeling)
df.drop(columns=['Employee ID'], inplace=True)

# Basic stats
print("\nDescriptive statistics:\n", df.describe())

# Create output directory for plots
os.makedirs("eda_outputs", exist_ok=True)

# Distribution of Burn Rate
plt.figure(figsize=(8, 4))
sns.histplot(df['Burn Rate'].dropna(), kde=True, color='orange')
plt.title("Distribution of Burn Rate")
plt.savefig("eda_outputs/burn_rate_distribution.png")
plt.close()

# Burn Rate vs Mental Fatigue Score
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df, x="Mental Fatigue Score", y="Burn Rate", hue="Gender")
plt.title("Burn Rate vs Mental Fatigue Score")
plt.savefig("eda_outputs/burnrate_vs_fatigue.png")
plt.close()

# Box plot of Burn Rate by Designation
plt.figure(figsize=(8, 4))
sns.boxplot(x='Designation', y='Burn Rate', data=df)
plt.title("Burn Rate by Designation")
plt.savefig("eda_outputs/burnrate_by_designation.png")
plt.close()

# Burn Rate by Company Type
plt.figure(figsize=(8, 4))
sns.boxplot(x='Company Type', y='Burn Rate', data=df)
plt.title("Burn Rate by Company Type")
plt.savefig("eda_outputs/burnrate_by_company_type.png")
plt.close()

# Heatmap for correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

print("EDA plots saved in 'eda_outputs/' folder.")
