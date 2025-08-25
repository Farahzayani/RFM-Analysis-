import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("Database/rfm_values.csv")

#Segmentation using K-Means 
#copy the data to apply transformation 
rfm_log = df[['Recency','Frequency','Monetary']].copy()

#rfm values are highly skewed ==> it's not normal distribution (not gaussian) so we need to normalize them for the next step (standarization)
#So we apply log() to compress large values and make distribution closer to normal (symmetric)
#apply log(1+x) instead of log(x) to avoid the case that log(1+0)=0 
rfm_log['Recency']= np.log1p(rfm_log['Recency'])
rfm_log['Frequency']= np.log1p(rfm_log['Frequency'])
rfm_log['Monetary']= np.log1p(rfm_log['Monetary'])

#standarization 
scaler= StandardScaler()   #standirazes features by removing the mean and divide by standard deviation (gaussien distribution parameter ) 
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled_df=pd.DataFrame (rfm_scaled , columns = ['Recency','Frequency','Monetary'], index = df.index)
print(rfm_scaled_df )


#search K of K-Means method 
#Elbow method ==> optimal K 
inertia =[]  #to stock sum of square distance between each point and its centroid FOR EACH K 
silhouette_scores = []  #coefficient between -1 and 1: how close each point of a one cluster to its neighbor cluster 
k_range = range (2,11)  #test of 2 to 10 clusters 

for k in k_range :
    kmeans=KMeans (n_clusters=k , random_state=42 , n_init = 'auto')
    kmeans.fit(rfm_scaled_df)   #Kmeans uses euclidean distance, that's why we standarized the data 
    inertia.append(kmeans.inertia_)

    if k>1 :  #that score needs at least 2 clusters to calculate the coefficient 
        silhouette_scores.append(silhouette_scores(rfm_scaled_df,kmeans.labels_))
    else :
        silhouette_scores.append(0)
