"""增强模型"""
import arrow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

filename_features="datas/features.csv"
filename_targets="datas/targets.csv"

"""获取模型评估指标"""
def get_reports(y_true,y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)  # 准确率
    precision = metrics.precision_score(y_true, y_pred) # 精确率
    recall = metrics.recall_score(y_true, y_pred)  # 召回率
    f1_score = metrics.f1_score(y_true, y_pred)    # f1 score
    metrics_scores = {"acc": acc, "pre": precision, "recall": recall, "f1": f1_score}
    return metrics_scores

def save_model(classifier):
    cTime = arrow.now()
    filename = 'pkls/ensemble_' + cTime.format('YYYY-MM-DD') + ' ' + str(cTime.hour) + '-' + str(
        cTime.minute) + '.pkl'  # 记录生成文件的时间
    joblib.dump(classifier, filename)


"""Adaboost (boost类)"""
def Adaboost():
    """模型参数设置"""

    # 弱学习器（决策树）
    baseClassifier=DecisionTreeClassifier(max_features=None,max_depth=100,min_samples_split=10,min_samples_leaf=10)
    adaClassifier=AdaBoostClassifier(base_estimator=baseClassifier,n_estimators=50,learning_rate=0.5)

    startTime = arrow.now()
    print("start at {}".format(startTime))

    # 分割数据集
    features = pd.read_csv("datas/features.csv", header=None).values  # pandas 转 numpy
    targets = pd.read_csv("datas/targets.csv", header=None).values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=35)
    print("data split finised!")

    adaClassifier.fit(X_train,y_train)
    #save_model(adaClassifier)

    y_pred=adaClassifier.predict(X_test)

    acc=metrics.accuracy_score(y_test,y_pred)
    reports=metrics.classification_report(y_test,y_pred)
    print("Finished!!   Running time:{}\n\n".format(arrow.now()-startTime))
    print("Accuracy:{}\n\n".format(acc))
    print(reports)

    metrics_scores=get_reports(y_test,y_pred)
    runTime=arrow.now()-startTime
    return metrics_scores,runTime



"""梯度提升树 (boost类)"""
def GBDT():
    pass

"""随机森林 (bagging类)"""
def RandomForests():
    pass

if __name__ == "__main__":
    Adaboost()