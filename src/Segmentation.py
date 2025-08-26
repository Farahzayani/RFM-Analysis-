import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score


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
        silhouette_scores.append(silhouette_score(rfm_scaled_df,kmeans.labels_))
    else :
        silhouette_scores.append(0)

# Visualisation of elbow method 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(k_range, inertia, 'bo-')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

#visualization of silhouette score method 
ax2.plot(k_range[1:], silhouette_scores[1:], 'ro-')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette score')
ax2.set_title('Silhouette score')
ax2.grid(True)

plt.tight_layout()
plt.show()

#the elbow (the point where the curve bends ) indicates the optimal K
#the K with the highest silhouette score is usually the best choice 
optimal_k = 4 

# Application of K-Means 
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(rfm_scaled_df)
df['KMeans_Cluster'] = kmeans.labels_   #add cluster labels to original rfm data 
# Analyze size of each cluster
cluster_sizes = df['KMeans_Cluster'].value_counts().sort_index()
print("Size of clusters:")
print(cluster_sizes)

# Calculate centroids of each cluster 
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), 
                           columns=['Recency', 'Frequency', 'Monetary'])

# Convert transformed data into original data using exp function (inverse of log ) 
centroids_df['Recency'] = np.expm1(centroids_df['Recency'])
centroids_df['Frequency'] = np.expm1(centroids_df['Frequency'])
centroids_df['Monetary'] = np.expm1(centroids_df['Monetary'])
print(centroids_df.round(2))

cluster_means = df.groupby('KMeans_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()  #ranking of each metric 

recency_rank = cluster_means['Recency'].rank(ascending=True)  #Higher score for lower recency
frequency_rank = cluster_means['Frequency'].rank(ascending=False)  #Higher score for higher frequency
monetary_rank = cluster_means['Monetary'].rank(ascending=False)  #Higher score for higher monetary value 

#put these scores into dict
cluster_to_r_score = recency_rank.astype(int).to_dict()
cluster_to_f_score = frequency_rank.astype(int).to_dict()
cluster_to_m_score = monetary_rank.astype(int).to_dict()

#apply these scores to each client 
df['R_Score'] = df['KMeans_Cluster'].map(cluster_to_r_score)
df['F_Score'] = df['KMeans_Cluster'].map(cluster_to_f_score)
df['M_Score'] = df['KMeans_Cluster'].map(cluster_to_m_score)

# Score global pondéré 
df['RFM_Global_Score'] = (df['R_Score']  * 0.5 +    #the most important factor ==> 50% of it participate in global rfm score
                           df['F_Score'] * 0.3 +    #important more then monetary so we take 30% of it 
                           df['M_Score'] * 0.2)     #it doesn't always guarantee future activity so just 20% (lowest height) 


df.to_csv ('Database/rfm_scores.csv', index=False, encoding='utf-8')