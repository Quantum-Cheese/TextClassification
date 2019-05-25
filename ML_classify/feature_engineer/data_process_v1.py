"""tfidf特征处理 + 降维"""

import jieba
jieba.load_userdict('myDict_20190326.txt')
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import arrow
from sklearn.decomposition import PCA
from dataObj import DataObj


data_filename="datas/raw_data_withTag.txt"
segament_filename = "datas/raw_features.txt"
feature_filename= "datas/features.csv"
exFeature_filename="datas/exFeatures.csv"

url= "192.168.20.149"
username="root"
password="admin123!@#"
db="text_classification_samples"


"""数据预处理：分词"""
def pre_process(data_filename,segament_filename):
    textList = []
    # 读取文本内容
    f = open(data_filename, encoding='utf-8')
    while True:
        line = f.readline() #逐行读取
        textList.append(line)
        if not line:
            break
    textList = textList[0:-1]
    f.close()

    ## 分词
    f1 = open(segament_filename, 'a', encoding='utf-8')
    for content in textList:
        # 把每篇的文章内容进行分词
        word_cut = pseg.cut(content)
        wordList = []
        for word, flag in word_cut:
            good_word = True
            # 去空字符
            if len(word) < 2:
                good_word = False
            # 词性选择
            if flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':  # 去除：人名/地名/机构团名/专有名词/时间词/副词
                good_word = False
            # 去除非中文字符
            for w in word:
                if w.encode('UTF-8').isalpha() or w.encode('UTF-8').isdigit():
                    good_word = False
            if good_word:
                wordList.append(word)
        # 把每篇文章的词列表重新组合成一个长字符串，词语中间用空格分隔
        long_s = (" ").join(wordList)
        # 把分词后的文章存入 txt文件，每篇文章写入一次（累加）
        f1.write(long_s)
        f1.write("\n")
    f1.close()


"""tfidf 处理"""
def feature_vectorization():

    startTime=arrow.now()

    # 词频统计并向量化(TF-IDF)
    f2=open(segament_filename,encoding='utf-8')
    lst=[]
    while True:
        line=f2.readline()
        lst.append(line)
        if not line:
            break
    f2.close()
    # print(len(lst))
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(lst[0:-1]).toarray()   # 去掉最后的空字符串（原始txt文件中的空行）
    # print(features)
    df1=pd.DataFrame(features)
    # 处理完成的特征向量存入csv
    df1.to_csv(feature_filename, header=False, index=False)

    runTime = arrow.now() - startTime
    print("data vectorization finished,running time:{}".format(runTime))


"""特征提取/降维"""
def feature_extraction(n):
    start_time=arrow.now()
    ## PCA 降维
    pca = PCA(n_components=n)

    df=pd.read_csv(feature_filename)

    exFeatures=pca.fit_transform(df.values())

    #降维后的特征向量再存入另一个csv
    df1=pd.DataFrame(exFeatures)
    df1.to_csv(exFeature_filename,header=False,index=False)

    print("feature extraction finished,running time:{}".format(arrow.now()-start_time))


if __name__ == "__main__":

    startTime=arrow.now()

    # 读取数据，存txt文件
    dataObj = DataObj(url, username, password, db)
    # 设定取样数量
    pos_samples,ratio=dataObj.get_positive_num(),1

    #pos_samples,ratio=100,1

    sql = "(SELECT title,content,information_type FROM samples_for_analysis where information_type=1) union all " \
          "(SELECT title,content,information_type FROM samples_for_analysis where information_type=0 order by rand() limit " \
          + str(pos_samples * ratio) + ")"
    print("Sample numbers:{}".format(pos_samples * (1 + ratio)))

    # 设定数据存储路径
    dataObj.txt_filename = "datas/raw_data_withTag.txt"

    dataObj.save_txt(sql)

    # 分词:读取原始数据文件，分词完后存到另一个文件中
    pre_process(data_filename,segament_filename)

    #分词后进行 tfidf 处理（向量化），输出该段运行时间
    feature_vectorization()

    #把tfidf处理后的稀疏向量进行降维（特征选择）
    feature_extraction(2000)

    # 统计总共运行时间
    totalTime = arrow.now() - startTime
    print("Total running time:{}".format(totalTime))








