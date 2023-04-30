

from sklearn.impute import SimpleImputer # boş değeleri doldurmak için
import pandas as pd

from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
# veriyi okuma
veri = pd.read_csv("maaslar.csv")

x = veri.iloc[:,1:2]
y = veri.iloc[:,2:]

X = x.values
Y = y.values
# lineer regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X))
plt.show()
# polinomal regresyon

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree= 5)
X_poly = poly.fit_transform(X)
print(X_poly)

lin2 = LinearRegression()
lin2.fit(X_poly, Y)
plt.scatter(X,Y)
plt.plot(X,lin2.predict(poly.fit_transform(X)),color ="red")
plt.show()


yeni_veri = lin_reg.predict([[11]])
yeni1 = lin_reg.predict([[6.6]])
print(yeni_veri)
print(yeni1)

print("------------------------------------------------------")
yeni_pol = lin2.predict(poly.fit_transform([[11]]))
yenipol1 =  lin2.predict(poly.fit_transform([[6.6]]))
print(yeni_pol)
print(yenipol1)



from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))





print("polinomal r2")
print(r2_score(Y, lin_reg.predict(poly.fit_transform(X))))












