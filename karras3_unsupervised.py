# CS 482 - Assignment 3
# Author: Demetri Karras
# File: karras3_unsupervised.py

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

""" ******** Load and Scale Data ******** """

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

""" ******** K-Means ******** """

kmeans = KMeans(n_clusters=3)

kmeans_y_pred = kmeans.fit_predict(X_scaled)

kmeans_ari = round(adjusted_rand_score(y, kmeans_y_pred), 3)
kmeans_silhouette = round(silhouette_score(X_scaled, kmeans_y_pred), 3)

print(" -- K-Means Metrics -- \n")
print(f"K-Means ARI: {kmeans_ari}")
print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print()

""" ******** Agglomerative Clustering ******** """

agglomerative = AgglomerativeClustering(n_clusters=3)

agglo_y_pred = agglomerative.fit_predict(X_scaled)

agglo_ari = round(adjusted_rand_score(y, agglo_y_pred), 3)
agglo_silhouette = round(silhouette_score(X_scaled, agglo_y_pred), 3)

print(" -- Agglomerative Clustering Metrics -- \n")
print(f"Agglomerative Clustering ARI: {agglo_ari}")
print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette}")
print()

""" ******** DBScan ******** """

# iterate through possible epsilon values until reaching 3 clusters
best_eps = -1
for eps in np.arange(0.1, 2.0, 0.1):

    dbscan = DBSCAN(eps=eps)
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels) # get rid of duplicate labels
    unique_labels.discard(-1) # remove noise

    num_clusters = len(unique_labels)

    # break loop when three clusters is reached
    if num_clusters == 3:
        best_eps = round(eps, 1)
        break

if best_eps != -1:

    dbscan = DBSCAN(eps=best_eps)
    dbscan_y_pred = dbscan.fit_predict(X_scaled)

    mask = dbscan_y_pred != -1

    dbscan_ari = round(adjusted_rand_score(y, dbscan_y_pred), 3)
    dbscan_silhouette = round(silhouette_score(X_scaled[mask], dbscan_y_pred[mask]), 3)

    print(" -- DBScan Metrics -- \n")
    print(f"Best Epsilon: {best_eps}")
    print(f"DBScan ARI: {dbscan_ari}")
    print(f"DBScan Silhouette Score: {dbscan_silhouette}")

else:

    print("DBScan was unable to produce 3 clusters.")
