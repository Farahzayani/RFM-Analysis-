from datetime import datetime 
import pandas as pd 
import matplotlib.pyplot as plt


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

#visualisations about rfm values 
plt.figure(figsize=(10, 6))
plt.bar (df['Recency'].unique(), df['Recency'].value_counts().values)
plt.xlabel('recency')
plt.ylabel('count')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter([x for x in range(len(df['Monetary']))] ,df['Monetary'] , color='g' )
plt.xlabel('Customers')
plt.ylabel('Tolat Amount') 
plt.title('Amount of purchases per customer')
plt.show()

