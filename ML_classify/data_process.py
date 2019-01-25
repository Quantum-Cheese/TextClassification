"""文本数据预处理"""
import pymysql
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

"""数据读取"""
def data_read(sql,url,username,password,dbname):
    conn = pymysql.connect(host=url, user=username, passwd=password, db=dbname, charset='utf8')
    curs = conn.cursor()
    curs.execute(sql)
    result = curs.fetchall()
    conn.close()
    return result


"""分词&预处理"""
def pre_process(textList):
    raw_featrues=[]
    targets=[]
    for title,content,type in textList:
        # 存储每篇文章的标签
        targets.append(type)
        # 把每篇的文章内容进行分词
        word_cut = pseg.cut(content)
        wordList = []
        for word, flag in word_cut:
            good_word = True
            if len(word) < 2:
                good_word = False
            for w in word:
                if w.encode('UTF-8').isalpha() or w.encode('UTF-8').isdigit():
                    good_word = False
            if good_word:
                wordList.append(word)
        # 把每篇文章的词列表重新组合成一个长字符串，词语中间用空格分隔
        long_s=(" ").join(wordList)
        raw_featrues.append(long_s)
    return raw_featrues,targets


"""特征提取(TF-IDF)"""
def feature_extraction(raw_features):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(raw_features).toarray()
    feature_names=vectorizer.get_feature_names()
    return feature_names,features


def data_split(features,targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    data_set=[X_train, X_test, y_train, y_test]
    return data_set



url="192.168.20.42"
username="root"
password="123456"
db="les_rulelib"
sql = "select title,content,information_type from les_cx_data_samples_processed limit 1,5"
contentList=data_read(sql,url,username,password,db)

raw_features,targets=pre_process(contentList)

feature_names,features=feature_extraction(raw_features)

data_set=data_split(features,targets)

