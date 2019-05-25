import pandas as pd
import arrow
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from dataObj import DataObj
import jieba
jieba.load_userdict('datas/myDict.txt')
import jieba.analyse
import jieba.posseg as pseg


rawDatas_filename="datas/raw_datas_v2.csv"
features_filename_1="datas/features_v2_1.csv"
features_filename_2="datas/features_v2_2.csv"
targets_filename="datas/targets_v2.csv"

def jieba_process(text):

    # jieba分词根据词性排除：人名/地名/机构团名/专有名词/时间词/副词/介词/连词/副动词/感叹词/.....
    stop_flags = ['nr', 'ns', 'nt', 'nz', 'p', 'q', 'r', 't', 'tg', 'vd', 'd', 'e', 'f', 'c', 'nrt']

    # 除去特殊字符和英文，保留汉字和数字
    text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039])", "", text)

    word_cut = pseg.cut(text)
    wordList = []
    for word, flag in word_cut:
        good_word = True
        if len(word) < 2 or word == "\r\n":
            good_word = False
        if flag in stop_flags:
            good_word = False
        if good_word:
            wordList.append(word)
    segment = (" ").join(wordList)  # 分词后重新合并成长字符串

    if len(segment)<2:
        segment="无用"

    # 基于tf-idf 抽取关键词，按照权重大小倒序排列(自动排除权重过低的词)
    keywords = jieba.analyse.extract_tags(segment, topK=len(wordList), withWeight=True, allowPOS=())
    new_text = ""
    for item in keywords:
        new_text += item[0] + " "

    return new_text


def segment(f_rawdatas):
    startTime = arrow.now()
    df = pd.read_csv(f_rawdatas)
    # 对每一行（每篇文章进行分词操作后,添加到新列‘segment_text’中）
    df['segment_text']=df['text'].apply(jieba_process)

    # 除去无用样本
    df=df[~df['segment_text'].isin(['无用'])]

    # 保存csv，覆盖原文件
    df.to_csv(f_rawdatas, index=False)
    print("Texts segment finished! Using time:{}".format(arrow.now() - startTime))


def tfidf_process(f_rawdatas,f_features,f_targets,tag=True):
    startTime = arrow.now()

    df=pd.read_csv(f_rawdatas)
    segment_list=df['segment_text'].values

    # 判断数据是否有标签，如果有就单独保存标签列
    if tag:
        df1 = pd.DataFrame(df['tag'])
        df1.to_csv(f_targets, index=False, header=False)

    # if-idf向量化
    # 设置df词频阈值，排除那些出现频率（逆文档频率）太高和太低的词
    vectorizer = TfidfVectorizer(max_df=0.8,min_df=0.05)  #筛选掉 80%以上的文档都出现的词 和 只有少于5%的文档才会出现的词
    features=vectorizer.fit_transform(segment_list).toarray()

    # 特征存入csv文件
    df1=pd.DataFrame(features)
    df1.to_csv(f_features,header=False,index=False)

    print("TTF-IDF feature vectorization finished! Using time:{}".format(arrow.now() - startTime))



def PCA_reducation():
    pass

def LDA_resucation():
    pass


if __name__ == "__main__":

    start_time=arrow.now()

    url = "192.168.20.149"
    username = "root"
    password = "admin123!@#"
    db = "text_classification_samples"
    dataObj = DataObj(url, username, password, db)

    pos_num=16000
    sql = "(SELECT title,content,information_type FROM samples_for_analysis where information_type=1 limit 8000) union all " \
          "(SELECT title,content,information_type FROM samples_for_analysis where information_type=0 limit 8000)" \

    print("Sample numbers:{}".format(pos_num))

    """读取数据，存csv"""
    dataObj.save_csv(sql, rawDatas_filename)
    """分词，tfidf向量化"""
    segment(rawDatas_filename)
    tfidf_process(rawDatas_filename,features_filename_1,targets_filename)  # 有标签

    # 查看特征向量大小
    df=pd.read_csv(features_filename_1)
    print("feature vector size after TF-IDF process:",df.shape)
