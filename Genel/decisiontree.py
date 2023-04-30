

from sklearn.impute import SimpleImputer # boş değeleri doldurmak için
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
# veriyi okuma
veri = pd.read_csv("maaslar.csv")

x = veri.iloc[:,1:2]
y = veri.iloc[:,2:]

X = x.values
Y = y.values




dt= DecisionTreeRegressor(random_state=0)
dt.fit(X,Y)
plt.scatter(X, Y)
plt.plot(X,dt.predict(X),color = "r")

print(dt.predict([[10]]))



from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators= 1000,random_state=0)
rfr.fit(X, Y.ravel())

print(rfr.predict([[10]]))

plt.scatter(X,Y)
plt.plot(X,rfr.predict(X))

from sklearn.metrics import r2_score
print("------------------------------")
print(r2_score(Y, rfr.predict(X)))