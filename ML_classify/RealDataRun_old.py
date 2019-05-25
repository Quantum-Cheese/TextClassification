import pandas as pd
import arrow
from sklearn.externals import joblib
from dataObj import DataObj
#from data_preprocess import segment,doc2vec
#from data_preprocess_v3 import segment,document_vectorization
from data_preprocess_v2 import segment,tfidf_process

rawDatas_filename="datas/testDatas.csv"
features_filename= "datas/features_test.csv"
result_filename="datas/test_prediction.csv"

"""数据读取"""
def data_read(startDate,endDate,rawDatas_filename):
    startTime=arrow.now()

    url = "192.168.20.149"
    username = "root"
    password = "admin123!@#"
    db = "text_classification_samples"
    data_obj = DataObj(url, username, password, db)
    #sql="select id,title,content from les_crawler_data_201903 where gmt_create> "+startDate+" and gmt_create < "+endDate+" limit 10"
    sql="select org_id,title,content from real_data_test where crawler_time >= "+startDate+" and crawler_time < "+endDate

    raw_datas=data_obj.data_read(sql)
    df = pd.DataFrame(data=[], columns=["org_id","text","tag"])  # text列包括标题和内容
    for i in range(0, len(raw_datas)):
        df.loc[i] = [raw_datas[i][0],raw_datas[i][1]+""+raw_datas[i][2],2]  # 合并标题和内容
    df.to_csv(rawDatas_filename,index=False)

    print("data saved into csv file. Using time:{}".format(arrow.now() - startTime))


"""预处理"""
def pre_process():
    startTime = arrow.now()

    segment(rawDatas_filename)

    #删除文本内容为空的数据
    df=pd.read_csv(rawDatas_filename)
    df.dropna(inplace=True)
    df.to_csv(rawDatas_filename,index=False)

    # word2vec，文档向量化
    #document_vectorization(rawDatas_filename,features_filename,"empty",tag=False)

    # tdidf 文档向量化
    tfidf_process(rawDatas_filename,features_filename,"empty",tag=False)

    print("Data pre_process finished.Using time:{}".format(arrow.now() - startTime))


"""模型预测并保存结果"""
def predict_to_csv(model_name):
    print(model_name)
    startTime=arrow.now()
    # 加载训练好的模型
    model=joblib.load(model_name)

    # 读取特征向量，预测
    features=pd.read_csv(features_filename,header=None).values
    prediction=model.predict(features)

    # 预测结果保存到新csv文件
    df=pd.read_csv(rawDatas_filename)
    new_df=pd.DataFrame(data=[], columns=["id","text","tag"])
    new_df['id']=df['org_id']
    new_df['text']=df['text']
    new_df['tag']=prediction

    # 去掉标签为0的，只保留预测为正例的
    new_df = new_df[~new_df['tag'].isin([0])]

    new_df.to_csv(result_filename,index=False)

    print('Prediction finished.Using time:{}'.format(arrow.now()-startTime))


def predict_to_mysql(model_name):
    pass


"""单次跑完所有待测数据"""
def single_run(model_name):
    startDate_1 = "'2019-03-18 00:00:00'"
    endDate_1 = "'2019-03-21 00:00:00'"

    # 读取待测数据（无标签）
    #data_read(startDate_1, endDate_1, rawDatas_filename)
    # 数据预处理
    pre_process()
    # 加载模型，预测并保存结果
    predict_to_csv(model_name)


"""拆分数据，分批跑预测"""
def batch_predict(model_name):
    features=pd.read_csv(features_filename,header=None)
    datas=pd.read_csv(rawDatas_filename)
    print(features.shape)
    print(datas.shape)

    # 存放预测结果的 df
    result_df = pd.DataFrame(data=[], columns=["id", "text", "tag"])

    # 数据分成四等份，分批预测
    ind_0,ind_1=0,24718
    for i in range(0,4):
        feature_df=features.iloc[ind_0:ind_1]
        data_df = datas.iloc[ind_0:ind_1]

        # 加载训练好的模型预测
        model = joblib.load(model_name)
        prediction = model.predict(feature_df)
        
        # 构造新df
        new_df=pd.DataFrame(data=[], columns=["id", "text", "tag"])
        new_df['id'],new_df['text']  = data_df['org_id'],data_df['text']
        new_df['tag'] = prediction

        #把这一批的预测结果加入 result_df 中
        result_df=pd.concat([result_df,new_df],axis=0,ignore_index=True)

        ind_0+=24718
        ind_1+=24718

    print(result_df.shape)

    # 去掉标签为0的，只保留预测为正例的
    result_df = result_df[~result_df['tag'].isin([0])]

    result_df.to_csv(result_filename, index=False)



if __name__=="__main__":

    model_name='pkls/GBDT_0415_1.pkl'
    single_run(model_name)









