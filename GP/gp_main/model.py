# 数据集上传不了github,提供百度网盘
# 百度网盘:链接: https://pan.baidu.com/s/1nD9ej_waIPpk_GITTbgXGA 密码: 3swf
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import time

train_data=pd.read_csv('./cnews.train.txt',sep='\t',names=['label','content'])
test_data=pd.read_csv('./cnews.test.txt',sep='\t',names=['content'])
# train_data.info()
# print(train_data.head())

def read_category(y_train):
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    label_id = []
    for i in range(len(y_train)):
        label_id.append(cat_to_id[y_train[i]])
    return label_id

train_target = train_data['label']
test_target = train_data['label']
y_train = read_category(train_target)
y_test = read_category(test_target)
# print(y_label_train)
# print(y_label_test)

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
# 不添加分词
#train_content = train_data['content']
#test_content = test_data['content']
# 添加分词
train_content =train_data['content'].apply(chinese_word_cut)
test_content = test_data['content'].apply(chinese_word_cut)

f_all = pd.concat(objs=[train_data['content'], test_data['content']], axis=0)
tfidf_vect = TfidfVectorizer(max_df = 0.9,min_df = 3,token_pattern=r"(?u)\b\w+\b")
tfidf_vect.fit(f_all)
X_train=tfidf_vect.fit_transform(train_data['content'])
X_test=tfidf_vect.fit_transform(test_data['content'])

# KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

# NB
nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test,y_pred))

# DT
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test,y_pred))

# SVM
svm = SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print(classification_report(y_test,y_pred))