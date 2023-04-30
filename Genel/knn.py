X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
predict = knn.predict(X)
print(predict)
print(knn.predict([[1.4]]))
print(knn.predict([[1.5]]))
print(knn.predict([[1.6]]))
