#coding=utf-8




from b2 import system2
system2.reload_utf8()
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# 交叉验证
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

import codecs


data = []
target = []
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
            data.append(content)
            target.append(score)
        except:
            continue
print len(data),len(target)
train_x , test_x , train_y , test_y = train_test_split(data,target,test_size = 0.25)


countvec = CountVectorizer(min_df = 2)
transformer = TfidfTransformer(use_idf=True)
model = LogisticRegression(C = 1e5 , max_iter = 300)
model = MultinomialNB()
clf_pipe = Pipeline([
    ('vectorizer',countvec),
    ('tfidf',transformer),
    ('classifieer',model)
]).fit(train_x,train_y)

for d,t in zip(test_x,test_y):
    print t,d.encode("utf-8"),clf_pipe.predict(d)


