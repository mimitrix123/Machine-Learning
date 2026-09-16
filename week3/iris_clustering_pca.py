"""Week 3: K-Means clustering, elbow method and PCA on Iris."""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.preprocessing import StandardScaler

iris=load_iris(); X=StandardScaler().fit_transform(iris.data); y=iris.target
inertia=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(X).inertia_ for k in range(1,9)]
plt.figure(); plt.plot(range(1,9),inertia,marker="o"); plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow Method"); plt.show()
km=KMeans(n_clusters=3,random_state=42,n_init=10); clusters=km.fit_predict(X)
print("Cluster centers:\n",km.cluster_centers_)
print("Silhouette score:",round(silhouette_score(X,clusters),4)); print("Adjusted Rand Index:",round(adjusted_rand_score(y,clusters),4))
pca=PCA(n_components=2); X2=pca.fit_transform(X); print("Explained variance ratio:",pca.explained_variance_ratio_); print("Total explained variance:",pca.explained_variance_ratio_.sum())
plt.figure(); plt.scatter(X2[:,0],X2[:,1],c=clusters,alpha=.8); c2=pca.transform(km.cluster_centers_); plt.scatter(c2[:,0],c2[:,1],marker="X",s=180); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Iris K-Means Clusters"); plt.show()
