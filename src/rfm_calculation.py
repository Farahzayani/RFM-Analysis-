from datetime import datetime 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_csv("Database/rfm_data.csv")

#convert all datetimes from object type into datetime type 
data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])
print (data['PurchaseDate'])

#count the number of occurrences of each unique customer (Since the rfm values will depend basically on CustomerID)
unique_customers = data['CustomerID'].value_counts()
print (unique_customers)
#calculate recency , frequency , monetary 

df=data.groupby('CustomerID').agg({
    'PurchaseDate':lambda x: (datetime.now().date()-x.max().date()).days ,  #days since the last purchase of a customer
    'OrderID' : 'count' ,                                                   #number of purchase per customer 
    'TransactionAmount':'sum' }                                             #the sum of amount of purchases per customer  
    )

#rename columns 
df = df.rename(columns = {'PurchaseDate':'Recency','OrderID':'Frequency','TransactionAmount':'Monetary'})
print (df.head())

#statistics of RFM values
print (df.describe()) 

#visualisations about rfm values :


# Distribution des métriques RFM
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

# Distribution de la Récence
sns.histplot(df['Recency'], bins=30, kde=True, ax=axes[0], color='cornflowerblue')  #kde=True : Adds a KDE curve (Kernel Density Estimate) on top of the histogram
axes[0].set_title('Recency Distribution')
axes[0].set_xlabel('Days since last purchase')

# Distribution de la Fréquence
sns.histplot(df['Frequency'], bins=30, kde=True, ax=axes[1], color='blue')
axes[1].set_title('Frequency Distribution')
axes[1].set_xlabel('Number of purchases')

# Distribution de la Valeur Monétaire
sns.histplot(df['Monetary'], bins=30, kde=True, ax=axes[2], color='darkblue')
axes[2].set_title('Monetary Distribution')
axes[2].set_xlabel('Total amount of purchases')

plt.tight_layout()  #automatically adjusts the spacing between subplots
plt.show()

#Boxplot to detect outliers 
plt.figure(figsize=(4, 5))
sns.boxplot(y=df['Monetary'])
plt.title('Monetary - Boxplot')
plt.show()
