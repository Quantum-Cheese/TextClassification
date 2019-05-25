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

# 定义输入数据（特征和标签）文件名
features_filaname="datas/features_0516.csv"
targets_filanme="datas/targets_0516.csv"


def data_split():
    features = pd.read_csv(features_filaname, header=None).values  # pandas 转 numpy
    targets = pd.read_csv(targets_filanme, header=None).values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=20)
    print("data split finised!")
    return (X_train, X_test, y_train, y_test)


"""获取模型评估指标"""
def get_reports(y_true,y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)  # 准确率
    precision = metrics.precision_score(y_true, y_pred) # 精确率
    recall = metrics.recall_score(y_true, y_pred)  # 召回率
    f1_score = metrics.f1_score(y_true, y_pred)    # f1 score
    metrics_scores = {"acc": acc, "pre": precision, "recall": recall, "f1": f1_score}
    return metrics_scores

def save_model(classifier,clf_name):
    cTime = arrow.now()

    filename = 'pkls/'+clf_name+"_"+ cTime.format('YYYY-MM-DD') + ' ' + str(cTime.hour) + '-' + str(
        cTime.minute) + '.pkl'  # 记录生成文件的时间
    joblib.dump(classifier, filename)

    model=joblib.load(filename)
    print(model)


"""Adaboost：手动调参"""
def Adaboost():
    """模型参数设置"""

    # 弱学习器（决策树）
    baseClassifier=DecisionTreeClassifier(max_features=None,max_depth=100,min_samples_split=10,min_samples_leaf=10)
    adaClassifier=AdaBoostClassifier(base_estimator=baseClassifier,n_estimators=150,learning_rate=0.1)

    startTime = arrow.now()
    print("start at {}".format(startTime))

    # 分割数据集
    X_train, X_test, y_train, y_test = data_split()


    adaClassifier.fit(X_train,y_train)
    save_model(adaClassifier,'Adaboost') #保存模型

    y_pred=adaClassifier.predict(X_test)

    print("accuracy:{}".format(metrics.accuracy_score(y_test,y_pred)))
    print("classification reports:{}".format(metrics.classification_report(y_test,y_pred)))
    print("Running time:{}".format(arrow.now()-startTime))

    metrics_scores=get_reports(y_test,y_pred)
    runTime=arrow.now()-startTime
    return metrics_scores,runTime


"""Adaboost：网格搜索调参"""
def Adaboost_cv():
    # 分割数据集
    X_train, X_test, y_train, y_test = data_split()

    params={
        "base_estimator__max_depth":[25,50],
        "base_estimator__min_samples_split":[20,40,60],
        "base_estimator__min_samples_leaf":[20,40,60],
        }

    DT=DecisionTreeClassifier(max_features=None)
    Adaboost_clf=AdaBoostClassifier(base_estimator=DT,learning_rate=1,n_estimators=50)

    grid_search_clf=GridSearchCV(Adaboost_clf,param_grid=params,cv=4)

    grid_search_clf.fit(X_train,y_train)
    best_clf=grid_search_clf.best_estimator_

    y_pred=best_clf.predict(X_test)

    acc=metrics.accuracy_score(y_test,y_pred)
    clf_reports=metrics.classification_report(y_test,y_pred)
    print(best_clf)
    print("Accuracy:{}".format(acc))
    print(clf_reports)



"""梯度提升树 GBDT"""
def GBDT():
    startTime=arrow.now()
    X_train, X_test, y_train, y_test = data_split()

    GBDT_clf=GradientBoostingClassifier(n_estimators=90,learning_rate=0.1,max_depth=10,max_features='sqrt',
                                        min_samples_split=400,min_samples_leaf=200,random_state=10,subsample=0.6)

    #GBDT_clf=GridSearchCV(estimator=base_clf,param_grid=params,cv=5,scoring='accuracy')
    #print("-------------\nRuning time:{}\n Best params:{}".format((arrow.now() - startTime),GBDT_clf.best_params_))

    GBDT_clf.fit(X_train,y_train)
    save_model(GBDT_clf,'GBDT')

    #clf = GBDT_clf.best_estimator_

    y_pred_0 = GBDT_clf.predict(X_train)  # 模型在训练集上做预测
    print("Scoring on training set:")
    print("Accuracy:{}\n Roc_auc_score:{}".format(metrics.accuracy_score(y_train,y_pred_0),metrics.roc_auc_score(y_train,y_pred_0)))
    print("Classification reports:{}".format(metrics.classification_report(y_train, y_pred_0)))
    print("-------------------------------------")

    y_pred=GBDT_clf.predict(X_test)  # 模型在测试集上做预测
    print("Accuracy:{}\n Roc_auc scores:{}".format(metrics.accuracy_score(y_test, y_pred),metrics.roc_auc_score(y_test, y_pred)))
    print("Classification reports:{}".format(metrics.classification_report(y_test, y_pred)))


"""随机森林 RandomForests"""
def RandomForests():
    # 模型参数设置
    RF_clf=RandomForestClassifier(n_estimators=100,oob_score=True,max_depth=None,min_samples_leaf=1,min_samples_split=2)

    startTime = arrow.now()
    print("start at {}".format(startTime))

    # 分割数据集
    X_train, X_test, y_train, y_test = data_split()

    RF_clf.fit(X_train,y_train)
    save_model(RF_clf,'RandomForests') #保存模型

    y_pred = RF_clf.predict(X_test)

    print("accuracy:{}".format(metrics.accuracy_score(y_test, y_pred)))
    print("classification reports:{}".format(metrics.classification_report(y_test, y_pred)))
    print("Running time:{}".format(arrow.now() - startTime))




if __name__ == "__main__":
    GBDT()
    #Adaboost_cv()
    #RandomForests()

