import numpy as np
from sklearn.linear_model import LinearRegression

a = [10,20,30,40,50,60]
b = [3,6,9,12,15,18]

x = np.array(a).reshape((-1,1))  
y = np.array(b).reshape((-1,1))

regr = LinearRegression().fit(x,y)

pred =regr.predict(np.array([90]).reshape((-1,1)))

print('Prediction for x =90, y= ' ,pred)
print('Coefficient = ' ,regr.coef_)
print('Intercept = ' , regr.intercept_)
