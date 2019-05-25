import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import jieba.posseg as pseg
import arrow


startTime=arrow.now()
"""向量化"""
count_vector = CountVectorizer()
documents=["我们","一起","出去","玩吧","哈哈"]
count_vector.fit(documents)
f_names=count_vector.get_feature_names()
# print(f_names)
doc_array = count_vector.transform(documents).toarray()
# print(type(doc_array))

orgData=(("浪莎智能家纺举报浪莎","1"),("商务部直销行业管理信息系统公布名单","0"),("有网友发来一份奖金制度","1"))
features=[]
targets=[]
for s,type in orgData:
    word_cut = pseg.cut(s)
    wordList = []
    for word, flag in word_cut:
        wordList.append(word)
    features.append(wordList)
    targets.append(type)
print(targets)
print(features)


"""特征提取"""

#预处理，数据格式转换
new_features=[]
for feature in features:
    # 转换成字符串，词之间用空格分隔
    s=(" ").join(feature)
    new_features.append(s)
# print(new_features)

"""方法一"""
# 用CountVectorizer
count_vector = CountVectorizer(analyzer='word')
count_vector.fit(new_features)
feature_vector=count_vector.transform(new_features).toarray()
feature_names=count_vector.get_feature_names()
# print(feature_names)
# print(feature_vector)
# 再用 TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(feature_vector)
# print(X_train_tfidf.toarray())

"""方法二"""
# 用TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(new_features)


print(vectorizer.get_feature_names())
print(X.toarray())

"""以上两种方法得出的结果一样"""

endTime=arrow.now()
print(endTime)
print(startTime)
print(endTime-startTime)

