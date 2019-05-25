"""用jieba预处理文本
分词+词性筛选+tfidf词频筛选
向量化之前对原始文本进行降维
"""

import jieba
import jieba.analyse
jieba.load_userdict("D:\TextClassification\ML_classify\myDict_20190326.txt")
import jieba.posseg as pseg
import pymysql

conn = pymysql.connect(host="192.168.20.149", user="root", passwd="admin123!@#", db="text_classification_samples", charset='utf8')
curs = conn.cursor()
sql="select content from samples_for_analysis where information_type=1 order by rand() limit 1"
curs.execute(sql)
result=curs.fetchall()

text=result[0][0]

stop_flags=['nr','ns','nt','nz','p','q','r','t','tg','vd','d','e','f','c','m','nrt']
# 去除：人名/地名/机构团名/专有名词/时间词/副词/介词/连词/副动词/感叹词/数词.....

# 先分词，词性选择，除去非法字符
word_cut = pseg.cut(text)
wordList = []
for word, flag in word_cut:
    good_word = True
    if len(word) < 2 or word == "\r\n":
        good_word = False
    if flag in stop_flags:
            good_word = False
    for w in word:
        if w.encode('UTF-8').isalpha() or w.encode('UTF-8').isdigit():
            good_word = False
    if good_word:
        wordList.append(word)
segment=(" ").join(wordList)  # 分词后重新合并成长字符串


# 基于tf-idf 抽取关键词，按照权重大小倒序排列(自动排除权重过低的词)
keywords = jieba.analyse.extract_tags(segment, topK=len(wordList),withWeight=True, allowPOS=())

new_text=""
for item in keywords:
    new_text+=item[0]+" "
    #print(item[0], item[1])  分别为关键词和相应的权重

print(new_text)






