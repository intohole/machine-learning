#coding=utf-8




from b2 import system2
system2.reload_utf8()
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# 交叉验证
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import codecs
from collections import defaultdict

data = []
target = []
labels = defaultdict(int)
with codecs.open("../../data/douban-comment.txt",encoding='utf-8',errors='ignore') as f:
    for line in f:
        line = line.strip().split("\t")
        if len(line) != 2:
            continue
        score , content = line
        if content is None or len(content) == 0:
            continue
        try:
            score = int(score) / 10
            labels[score] += 1
            data.append(content)
            target.append(score)
        except:
            continue
print labels
print len(data),len(target)
train_x , test_x , train_y , test_y = train_test_split(data,target,test_size = 0.25)


countvec = CountVectorizer(min_df = 2)
transformer = TfidfTransformer(use_idf=True)
model = OneVsRestClassifier(LinearSVC(random_state = 0))
clf_pipe = Pipeline([
    ('countvec',countvec),
    ('tfidf',transformer),
    ('classifieer',model)
]).fit(train_x,train_y)
model_y = clf_pipe.predict(test_x)
# 评价分类模型的， 有归一化选项，默认为true
print accuracy_score(test_y,model_y)

