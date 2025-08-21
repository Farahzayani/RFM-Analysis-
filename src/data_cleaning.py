import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Read dataset 
data = pd.read_csv("Database/rfm_data.csv")
print(data.head())

# Data Exploration 

print("Shape of Dataset")
print(f"Dimensions: {data.shape[0]} lignes, {data.shape[1]} columns")
print ("General Informations")
data.info()
print ("descriptive statistics")
print (data.describe())

# Missing values verification 
missing_values= data.isnull().sum()
missing_values_percent= (missing_values*100)/len(data)
print ("Missing values \n" , missing_values_percent)

# Duplicates verification 
print ("Duplicate values :", data.duplicated().sum())

# Frequency of purchases per customer 
plt.figure(figsize=(10, 6))
sns.histplot(data['CustomerID'].value_counts())
plt.title('Frequency of purchases per customer')
plt.xlabel('Number of purchases')
plt.ylabel('Number of customers')
plt.show()
print (data ['CustomerID'].value_counts()) 

# Verify outliers with quantiles 
def outliers (col,df):
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

for col in data.columns :
    outliers(col,data)

# Outliers detection with visualisation 
sns.set(style="whitegrid") #configure the default aesthetic style for all Seaborn plots
plt.figure(figsize=(10, 6)) 
sns.boxplot(x=data["TransactionAmount"])
plt.title("Transaction amount Distribution")
plt.show()