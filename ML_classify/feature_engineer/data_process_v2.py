"""doc2vec文档向量化(全程用csv文件操作)"""
import pandas as pd
import arrow
import jieba
jieba.load_userdict('myDict_20190326.txt')
import jieba.posseg as pseg
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dataObj import DataObj

filename_rawdatas="datas/raw_datas.csv"
filename_features="datas/features.csv"
filename_targets="datas/targets.csv"

""" 原始方案：遍历文本列表，先整体分词再整体保存（df生成新列）"""
def segment_jieba():
    startTime=arrow.now()
    df=pd.read_csv(filename_rawdatas)
    raw_texts = df.raw_text

    segment_texts=[]
    # 对每篇文本分词
    for text in raw_texts:
        word_cut = pseg.cut(text)
        wordList = []
        for word, flag in word_cut:
            good_word = True
            # 去掉非法字符（空字符，标点符号，换行符等）
            if len(word) < 2 or word=="\r\n":
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
        if len(wordList)<1:
            segment_text="无用"
        else:
            segment_text=(" ").join(wordList)

        segment_texts.append(segment_text)

    # 分词后的词列表作为新列添加到df中
    df['segment_text']=segment_texts

    # 分词后保存,覆盖原文件
    df.to_csv(filename_rawdatas,index=False)

    print("Texts segment finished! Using time:{}".format(arrow.now() - startTime))


def jieba(text):
    """
    对单篇文本进行jieba分词
    :param text: 待分词文本(长字符串）
    :return: 分词后的新文本（词语之间用空格分割）
    """
    word_cut = pseg.cut(text)
    wordList = []
    for word, flag in word_cut:
        good_word = True
        # 去掉非法字符（空字符，标点符号，换行符等）
        if len(word) < 2 or word == "\r\n":
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
    if len(wordList) < 1:
        segment_text = "无用"
    else:
        segment_text = (" ").join(wordList)

    return segment_text


def segment():
    startTime = arrow.now()

    df = pd.read_csv("datas/raw_datas.csv")
    # 对每一行（每篇文章进行分词操作后直接替换原文本）
    df['text']=df.text.apply(jieba)
    # 保存csv，覆盖原文件
    df.to_csv("datas/raw_datas.csv", index=False)

    print("Texts segment finished! Using time:{}".format(arrow.now() - startTime))



"""doc2vec向量化，处理后的特征向量存成新csv文件"""
def doc2vec():
    startTime=arrow.now()

    df=pd.read_csv("datas/raw_datas.csv")
    vector_num=df.shape[0]

    # 标签列单独保存
    df1=pd.DataFrame(df.tag)
    df1.to_csv("datas/targets.csv",index=False,header=False)

    # 构建2d词列表
    textList=[]
    for text in df['text']:
        try:
            textList.append(text.split(" "))
        except Exception:
            print(text)


    # 构建doc2vec模型训练的输入数据
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(textList)]

    # 训练文档（句子）向量模型#
    model = Doc2Vec(documents, dm=0, vector_size=100, window=8, min_count=1,
                    workers=6)  # 默认dm=1（PV-DM 模型）;dm=0 (PV-DBOW)
    model.save('doc_vecs.model')
    model = Doc2Vec.load('doc_vecs.model')

    ##  文档向量数据格式转换，存入scv

    # 先构造二维list
    vector_lists = [list(model.docvecs[n]) for n in range(0, vector_num)]
    # 转成pandas
    df = pd.DataFrame(vector_lists)
    # 存csv
    df.to_csv("datas/features.csv", header=False, index=False)

    print("Document vectorization finished! Using time:{}".format(arrow.now() - startTime))


if __name__ == "__main__":

    start_time=arrow.now()

    url = "192.168.20.149"
    username = "root"
    password = "admin123!@#"
    db = "text_classification_samples"
    dataObj = DataObj(url, username, password, db)

    # 设定取样数量
    pos_samples, ratio = dataObj.get_positive_num(), 1

    # sql = "(SELECT title,content,information_type FROM samples_for_analysis where information_type=1 limit " + str(
    #     pos_samples) + ") union all (SELECT title,content,information_type FROM samples_for_analysis where information_type=0 order by rand() limit " \
    #       + str(pos_samples * ratio) + ")"

    sql = "(SELECT title,content,information_type FROM samples_for_analysis where information_type=1) union all " \
          "(SELECT title,content,information_type FROM samples_for_analysis where information_type=0 order by rand() limit " \
          + str(pos_samples * ratio) + ")"
    print("Sample numbers:{}".format(pos_samples*(1+ratio)))

    """读取数据，存csv"""
    dataObj.save_csv(sql,filename_rawdatas)

    """分词，doc2vec向量化"""
    segment()
    doc2vec()

    print("Program finished!!Total running time:{}".format(arrow.now()-start_time))

