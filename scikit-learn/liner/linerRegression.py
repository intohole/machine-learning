#coding=utf-8





from sklearn import linear_model
import numpy as np
import random
# fit_intercept -> 截距
# normalize -> 是否归一化
liner = linear_model.LinearRegression(fit_intercept = False,normalize = True)
x = [[random.randint(0,3000) ,random.randint(0,3000) , random.randint(0,3000),random.randint(0,3000)] for i in range(10000)]

y = [ sum(j for j in i) for i in x]
liner.fit(x,y)
print liner.predict([20,30,20,10])
print liner.coef_
