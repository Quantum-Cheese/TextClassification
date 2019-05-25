from gensim.models import word2vec
import pandas as pd
import numpy as np
import pymysql
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile

def read_keyWords():
    conn = pymysql.connect(host="192.168.20.42", user="root", passwd="123456", db="les_rulelib",
                           charset='utf8')
    curs = conn.cursor()
    curs.execute("select term from les_cx_features")
    result = curs.fetchall()
    conn.close()

    return [r[0] for r in result]


model1 = word2vec.Word2Vec.load('word2vecs/w2v_1.model')
model1.wv.save_word2vec_format('word2vecs/w2v_cbow_1.txt',binary = False)

vocab=model1.wv.vocab
print(len(vocab))


target_words=read_keyWords()

"""训练好的word2vec模型，相似词测试（特征词表中每个特征词取前10个相似词）"""
datas=[]
# 构造2d lists data=[['主词1'，'近似词1']，[....],....]
n=0
while n <len(target_words):
    try:
        lst=[key[0] for key in model1.wv.similar_by_word(target_words[n], topn =10)]
        long_txt=(" ").join(lst)
        datas.append([target_words[n],long_txt])
    except:
        pass
    finally:
        n+=1

# df=pd.DataFrame(data=datas,columns=['main_word','similar_words'])
# df.to_csv("datas/w2v_1-simwords.csv",index=False)





