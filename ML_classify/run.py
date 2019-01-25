import data_process
from NBayes import model_NByes
from SVM import linear_SVC
from SVM import nonLinear_SVC
import arrow

start=arrow.now()
"""数据预处理"""
url="192.168.20.42"
username="root"
password="123456"
db="les_rulelib"
# 选择指定条数的正例和负例
sql = "SELECT title,content,information_type FROM `samples_for_anlysis` where information_type=1 " \
      "union all SELECT title,content,information_type from samples_for_anlysis where information_type=0 limit 10000;"
contentList=data_process.data_read(sql,url,username,password,db)
raw_features,targets=data_process.pre_process(contentList)
feature_names,features=data_process.feature_extraction(raw_features)
data_set=data_process.data_split(features,targets)


"""模型预测"""
# 1. 使用朴素贝叶斯（高斯）分类器
scores_nbyes,runTime_nbyes=model_NByes(data_set)

# 2. 线性SVM
scores_lsvc,runTime_lsvc=linear_SVC(data_set)

# 运行总时间
end=arrow.now()
total_time=end-start

# 3.非线性SVC
# scores_nsvc,runTime_nsvc=nonLinear_SVC(data_set)
# print("Using nonlinear SVC...............result:")
# print(scores_nsvc)

"""输出模型评估结果（导出文件）"""
f=open("results.txt","a")
f.write("Using Naive Bayes classifier......result:\n")
f.write("running time:{}\n".format(runTime_nbyes))
f.write("accuracy: {}  precision: {}  recall: {}   f1_score: {}"
        .format(scores_nbyes["acc"],scores_nbyes["pre"],scores_nbyes["recall"],scores_nbyes["f1"]))
f.write("\n\n")

f.write("Using linear SCV classifier......result:\n")
f.write("running time:{}\n".format(runTime_lsvc))
f.write("accuracy: {}   precision: {}   recall: {}  f1_score: {}"
        .format(scores_lsvc["acc"],scores_lsvc["pre"],scores_lsvc["recall"],scores_lsvc["f1"]))
f.write("\n\n")

f.write("Total running time:{}".format(total_time))

