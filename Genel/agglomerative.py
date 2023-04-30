import numpy as np
import matplotlib.pyplot as plt
# Create the data
x = np.random.randint(0, 100, (100, 2))
print(x)
print(x[:,0])
print("****************")
print(x[:,1])
# Plot the data
# plt.scatter(x[:, 0], x[:, 1])
#plt.show()



# Create the model
# hiyerarşik kümeleme
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_tahmin = model.fit_predict(x)
# Print the clusters
print(y_tahmin)
# Plot the clusters
#plt.scatter(x[y_tahmin==0, 0], x[y_tahmin==0, 1], c='red')
#plt.scatter(x[y_tahmin==1, 0], x[y_tahmin==1, 1], c='blue')
#plt.scatter(x[y_tahmin==2, 0], x[y_tahmin==2, 1], c='green')
#plt.show()
# Print the centroids

# Plot the centroids
import  scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()

