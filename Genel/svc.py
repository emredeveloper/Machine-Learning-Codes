from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC(kernel="rbf")
X = [[10, 15], [20, 30], [50, 70], [90, 100]]
y = [0, 1, 2, 3]
svc.fit(X,y)
y_pred = svc.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Doğru tahmin oranı: ", accuracy)
print(svc.predict([[20,75]]))
