#coding=utf-8





from sklearn import linear_model
import numpy as np
import random

# C -> regularization 正则化 对整体系数进行惩罚，避免过拟合
# max_iter 迭代次数
lr = linear_model.LogisticRegression(C = 1e5,max_iter = 1000)

x = [[random.randint(0,1000),random.randint(0,1000)] for i in range(2000)]

y = [ i[0] + i[1] for i in x]
lr.fit(x,y)
print lr.predict([20,30])
print lr.coef_
