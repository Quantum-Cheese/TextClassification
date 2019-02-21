"""支持向量机"""
from sklearn.model_selection import train_test_split
import arrow
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


"""获取模型评估指标"""
def get_reports(y_true,y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)  # 准确率
    precision = metrics.precision_score(y_true, y_pred) # 精确率
    recall = metrics.recall_score(y_true, y_pred)  # 召回率
    f1_score = metrics.f1_score(y_true, y_pred)    # f1 score
    metrics_scores = {"acc": acc, "pre": precision, "recall": recall, "f1": f1_score}
    return metrics_scores


"""线性内核(网格搜索）"""
def linear_SVC_grid():
    startTime = arrow.now()
    print("start at {}".format(startTime))

    # 分割数据集
    features = pd.read_csv("features.csv", header=None).values
    targets = pd.read_csv("targets.csv", header=None).values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=35)
    print("data split finised!")

    svc=LinearSVC()
    # 网格搜索，C参数
    params={'C':[1,20]}
    clf=GridSearchCV(svc,params,cv=8)
    clf.fit(X_train,y_train)
    best_clf=clf.best_estimator_
    # 保存模型
    joblib.dump(best_clf, 'model_lsvc.pkl')
    # 打印最佳参数
    print("Best classifier:{}".format(best_clf))
    y_pred=best_clf.predict(X_test)
    metrics_scores=get_reports(y_test,y_pred)

    # 运行时间
    endTime = arrow.now()
    run_time = endTime - startTime

    return metrics_scores,run_time

"""线性内核（固定参数）"""
def linear_svc_nogrid():
    startTime = arrow.now()
    print("start at {}".format(startTime))

    # 分割数据集
    features = pd.read_csv("features.csv", header=None).values
    targets = pd.read_csv("targets.csv", header=None).values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=35)
    print("data split finised!")

    svc = LinearSVC(random_state=0,C=5)
    svc.fit(X_train,y_train)
    joblib.dump(svc, 'model_lsvc_1.pkl')
    y_pred=svc.predict(X_test)
    scores=get_reports(y_test,y_pred)

    # 运行时间
    endTime = arrow.now()
    run_time = endTime - startTime

    return scores, run_time


"""非线性内核"""
def nonLinear_SVC():
    startTime = arrow.now()

    # 分割数据集
    features = pd.read_csv("features.csv", header=None).values
    targets = pd.read_csv("targets.csv", header=None).values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=35)
    print("data split finised!")

    svc=SVC()
    # 网格搜索：kernel，gamma，c参数
    params={"kernel":['poly', 'rbf'],"gamma":[0,1],"C":[1,10]}
    clf = GridSearchCV(svc, params, cv=5)
    clf.fit(X_train, y_train)
    best_clf = clf.best_estimator_
    # 保存模型
    joblib.dump(best_clf, 'model_lsvc.pkl')
    y_pred = best_clf.predict(X_test)
    metrics_scores = get_reports(y_test, y_pred)
    # 运行时间
    endTime = arrow.now()
    run_time = endTime - startTime

    return metrics_scores, run_time

# 运行固定参数模型
scores,runTime=linear_svc_nogrid()

# 运行网格搜索模型
#scores,runTime=linear_SVC_grid()

print(scores)
print(runTime)
print("program finished!!")

"""输出模型评估结果（导出文件）"""
f=open("results.txt","a")
f.write("Using Linear SVM classifier......result:\n")
f.write("running time:{}\n".format(runTime))
f.write("accuracy: {}  precision: {}  recall: {}   f1_score: {}"
        .format(scores["acc"],scores["pre"],scores["recall"],scores["f1"]))
f.write("\n\n")
f.close()
