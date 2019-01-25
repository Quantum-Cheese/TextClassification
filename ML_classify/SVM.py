"""支持向量机"""

import arrow
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import GridSearchCV




"""获取模型评估指标"""
def get_reports(y_true,y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)  # 准确率
    precision = metrics.precision_score(y_true, y_pred) # 精确率
    recall = metrics.recall_score(y_true, y_pred)  # 召回率
    f1_score = metrics.f1_score(y_true, y_pred)    # f1 score
    metrics_scores = {"acc": acc, "pre": precision, "recall": recall, "f1": f1_score}
    return metrics_scores


"""线性内核"""
def linear_SVC(data_set):
    startTime = arrow.now()

    X_train, X_test, y_train, y_test = data_set[0],data_set[1],data_set[2],data_set[3]
    svc=LinearSVC()
    # 网格搜索，C参数范围：1-10
    params={'C':[1,10]}
    clf=GridSearchCV(svc,params,cv=8)
    clf.fit(X_train,y_train)
    best_clf=clf.best_estimator_
    # 保存模型
    joblib.dump(best_clf, 'model_lsvc.pkl')
    y_pred=best_clf.predict(X_test)
    metrics_scores=get_reports(y_test,y_pred)

    # 运行时间
    endTime = arrow.now()
    run_time = endTime - startTime

    return metrics_scores,run_time


"""非线性内核"""
def nonLinear_SVC(data_set):
    startTime = arrow.now()
    X_train, X_test, y_train, y_test = data_set[0],data_set[1],data_set[2],data_set[3]
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
