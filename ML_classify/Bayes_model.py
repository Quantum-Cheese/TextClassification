"""朴素贝叶斯模型（高斯)"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import arrow
from sklearn.externals import joblib
import pandas as pd

def run_model():
    startTime = arrow.now()

    print("start at {}\n".format(startTime))

    features = pd.read_csv("feature_engineer/datas/features.csv", header=None).values
    targets = pd.read_csv("feature_engineer/datas/targets.csv", header=None).values.ravel()

    sample_number=len(targets)
    print("sample number:{}".format(sample_number))

    X_train, X_test, y_train, y_test=train_test_split(features,targets,test_size=0.2,random_state=35)
    print("data split finised!")

    """模型预测"""
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print("model training completed!!")
    # 保存模型
    cTime = arrow.now()
    filename = 'pkls/lsvc_' + cTime.format('YYYY-MM-DD') + ' ' + str(cTime.hour) + '-' + str(
        cTime.minute) + '.pkl'  # 记录生成文件的时间
    joblib.dump(clf, filename)

    y_pred = clf.predict(X_test)

    """效果评估"""
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    metrics_scores = {"acc": acc, "pre": precision, "recall": recall, "f1": f1_score}

    # 运行时间
    endTime = arrow.now()
    run_time = endTime - startTime

    return metrics_scores, run_time,sample_number


if __name__ == "__main__":
    # 加载特征工程处理好的数据，运行模型预测
    scores, runTime,sample_number = run_model()
    print(scores)
    print(runTime)
    print("program finished!!")

    """输出模型评估结果（导出文件）"""
    cTime = arrow.now()
    filename="results "+ cTime.format('YYYY-MM-DD') + ' ' + str(cTime.hour) + '-' + str(
        cTime.minute) +".txt"
    f = open(filename, "a")
    f.write("Training Naive Bayes classifier with {} samples.\n Result:\n".format(sample_number))
    f.write("running time:{}\n".format(runTime))
    f.write("accuracy: {}  precision: {}  recall: {}   f1_score: {}"
            .format(scores["acc"], scores["pre"], scores["recall"], scores["f1"]))
    f.write("\n\n")







