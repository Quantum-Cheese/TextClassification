"""文本数据预处理"""
import pymysql
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import arrow

"""数据读取"""
def data_read(sql,url,username,password,dbname):
    # 读取数据库数据
    conn = pymysql.connect(host=url, user=username, passwd=password, db=dbname, charset='utf8')
    curs = conn.cursor()
    curs.execute(sql)
    result = curs.fetchall()
    f=open("datas.txt","a",encoding="utf-8")
    targets=[]
    for content in result:
        # 把标题和内容合并，每篇文章按行分开
        s=content[0]+","+content[1]
        s1=s.replace("\n","")
        long_text=s1.replace("\r","")
        f.write(long_text)
        f.write("\n")
        # 单独存储标签
        targets.append(content[2])
    f.close()
    conn.close()
    df = pd.DataFrame(targets)
    # 把标签列表转成pandas,存csv
    df.to_csv("targets.csv", header=False, index=False)


"""特征预处理"""
def pre_process():
    textList=[]
    f = open("datas.txt", encoding='utf-8')
    while True:
        line = f.readline()
        textList.append(line)
        if not line:
            break
    textList=textList[0:-1]
    f.close()

    ## 分词
    f1=open("raw_features.txt",'a',encoding='utf-8')
    for content in textList:
        # 把每篇的文章内容进行分词
        word_cut = pseg.cut(content)
        wordList = []
        for word, flag in word_cut:
            good_word = True
            if len(word) < 2:
                good_word = False
            if flag=='nr' or flag=='ns' or flag=='nt' or flag=='nz':
                good_word = False
            for w in word:
                if w.encode('UTF-8').isalpha() or w.encode('UTF-8').isdigit():
                    good_word = False
            if good_word:
                wordList.append(word)
        # 把每篇文章的词列表重新组合成一个长字符串，词语中间用空格分隔
        long_s=(" ").join(wordList)
        # 把分词后的文章按行存入 txt文件
        f1.write(long_s)
        f1.write("\n")
    f1.close()

    ## 特征提取(TF-IDF)

    f2=open("raw_features.txt", encoding='utf-8')
    lst=[]
    while True:
        line=f2.readline()
        lst.append(line)
        if not line:
            break
    f2.close()
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(lst[0:-1]).toarray()
    df1=pd.DataFrame(features)
    df1.to_csv("features.csv", header=False, index=False)


startTime=arrow.now()

url="192.168.20.149"
username="root"
password="admin123!@#"
db="text_classification_samples"
sql = "SELECT title,content,information_type FROM samples_for_anlysis where information_type=1 union all " \
      "SELECT title,content,information_type from samples_for_anlysis where information_type=0 limit 20000"
data_read(sql,url,username,password,db)


curentTime1=arrow.now()-startTime
print("data reading finished,running time:{}".format(curentTime1))
pre_process()
curentTime=arrow.now()-curentTime1
print("data processing finished,running time:{}".format(curentTime))

totalTime=arrow.now()-startTime
print("Total running time:{}".format(totalTime))






