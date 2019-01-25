"""朴素贝叶斯模型（高斯)"""
import data_process
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import arrow
from sklearn.externals import joblib


def model_NByes(data_set):
    startTime=arrow.now()

    X_train, X_test, y_train, y_test = data_set[0],data_set[1],data_set[2],data_set[3]

    """模型预测"""
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    # 保存模型
    joblib.dump(clf, 'model_nbyes.pkl')
    clf_1=joblib.load('model_nbyes.pkl')
    y_pred = clf_1.predict(X_test)

    """效果评估"""
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall=metrics.recall_score(y_test,y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    metrics_scores={"acc":acc,"pre":precision,"recall":recall,"f1":f1_score}

    # 运行时间
    endTime=arrow.now()
    run_time=endTime-startTime

    return metrics_scores,run_time


