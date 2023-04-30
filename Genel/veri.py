# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:04:49 2023

@author: Cihat Emre Karataş
"""

from sklearn.impute import SimpleImputer # boş değeleri doldurmak için
import pandas as pd
import numpy as np
import matplotlib as plt
veri = pd.read_csv("eksikveriler.csv")

imputer = SimpleImputer(missing_values= np.nan,strategy= 'mean') # nan olanları ortalama değer ile doldurma
veri

Yas = veri.iloc[:,1:4].values
Yas

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4]) # transform eğittikten sonra değerleri yerleştirmek için kullanılıyor
Yas

ulkeler = veri.iloc[:,0:1].values
ulkeler

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulkeler[:,0] = le.fit_transform(ulkeler[:,0]) # bu kodlar ile metin değerlerini sayısal değerlere döndürdük 
ulkeler

ohe = preprocessing.OneHotEncoder()
ulkeler = ohe.fit_transform(ulkeler).toarray()
ulkeler

c = veri.iloc[:,-1:].values
c

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(c[:,-1]) # bu kodlar ile metin değerlerini sayısal değerlere döndürdük 
c

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
c


sonuc = pd.DataFrame(data= ulkeler,index = range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas,index = range(22),columns= ["Boy","Kilo","Yaş"])
print(sonuc2)

sonuc3 = pd.DataFrame(data= c,index= range(22),columns= ["E","K"])
print(sonuc3)

ulke_yas = pd.concat([sonuc,sonuc2],axis=1)
print(ulke_yas)

toplam_sonuc = pd.concat([ulke_yas,sonuc3],axis=1)
print(toplam_sonuc)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(ulke_yas,toplam_sonuc,test_size=0.33,random_state=0) # %33 teste ayırdık


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_predict = reg.predict(x_test)


boy = sonuc2.iloc[:,0].values
print(boy)
sol = toplam_sonuc.iloc[:,:3]
sag = toplam_sonuc.iloc[:,4:]
print(sol)
print(sag)

veri= pd.concat([sol,sag],axis = 1)


x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size= 0.33, random_state= 0)
model = LinearRegression()
model.fit(x_train,y_train)
tahmin = model.predict(x_test) # x_teste bakıp tahmini bir boy veriyor
print(tahmin)

"""
Bu kodda, statsmodels.api kütüphanesi kullanılarak bir OLS (En Küçük Kareler Yöntemi) modeli oluşturuluyor.

İlk olarak, np.ones((22,1)).astype(int) koduyla 22 satır ve 1 sütundan oluşan bir dizi oluşturuluyor ve bu diziye arr adı veriliyor. Bu dizi, bağımsız değişkenlerin katsayılarını hesaplamak için kullanılacak sabit terimi içeriyor.

Daha sonra, values= veri ile veri veri setinin değerleri values parametresine atanıyor. axis=1 parametresi, bu değerlerin satır bazında birleştirileceğini belirtiyor.

X_l satırı, veri setinden ilk 6 sütunun ([0,1,2,3,4,5]) verilerini içeren bir dizidir. Bu dizi ayrıca float türüne dönüştürülüyor.

sm.OLS(boy, X_l) ile OLS modeli oluşturuluyor ve fit() yöntemi ile model uygun hale getiriliyor. Bu modelin özet istatistikleri print(model.summary()) komutuyla ekrana yazdırılıyor.

P>|t| ifadesi, her bir bağımsız değişkenin t-testi için hesaplanan p-değeridir. Bu değer, bağımsız değişkenin modeldeki etkisinin anlamlılığını belirler. Anlamlılık düzeyi (significance level) genellikle 0.05 olarak belirlenir. Eğer bir bağımsız değişkenin p-değeri 0.05'ten küçükse, o değişkenin modeldeki etkisi anlamlıdır ve modelde yer alabilir.
"""
import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int),values= veri,axis = 1)
X_l = veri.iloc[:,[0,1,2,3,4,5]]
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X = np.append(arr = np.ones((22,1)).astype(int),values= veri,axis = 1)
X_l = veri.iloc[:,[0,1,2,3,4]]
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X = np.append(arr = np.ones((22,1)).astype(int),values= veri,axis = 1)
X_l = veri.iloc[:,[0,1,2,3]]
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

























