"""数据源对象"""
import pymysql
import pandas as pd
import arrow

class DataObj():

    def __init__(self,url,username,password,dbname):
        self.url=url
        self.username=username
        self.password=password
        self.dbname=dbname
        self.rawData_filename="datas/raw_data_withTag.txt"
        self.targets_filename="datas/targets.csv"
        self.sql=""

    def get_positive_num(self):
        conn = pymysql.connect(host=self.url, user=self.username, passwd=self.password, db=self.dbname, charset='utf8')
        curs = conn.cursor()
        sql="select count(id) from samples_for_analysis where information_type=1"
        curs.execute(sql)
        result=curs.fetchall()
        return result[0][0]


    def set_sql(self,positive_num,ratio):
        """
        :param positive_num:  正例数量
        :param ratio:  负例/正例 比例（必须是整数）
        """

        # 负例随机选取
        sql="(SELECT title,content,information_type FROM samples_for_analysis where information_type=1 limit "+str(positive_num)+") " \
              "union all (SELECT title,content,information_type FROM samples_for_analysis where information_type=0 order by rand() limit " \
              +str(positive_num*ratio)+")"

        self.sql=sql

    def data_read(self,sql):
        startTime = arrow.now()
        ## 读取数据库数据
        conn = pymysql.connect(host=self.url, user=self.username, passwd=self.password, db=self.dbname, charset='utf8')
        curs = conn.cursor()
        curs.execute(sql)
        result = curs.fetchall()
        conn.close()
        runTime = arrow.now() - startTime
        # 输出数据读取时间
        return result

    def save_txt(self,sql):
        startTime=arrow.now()
        result=self.data_read(sql)
        f = open("datas/raw_data_withTag.txt", "a", encoding="utf-8")
        targets = []
        ## 数据存入硬盘，文本内容写入txt文件，标签存为csv
        for content in result:
            # 把标题和内容合并，每篇文章按行分开
            s = content[0] + "," + content[1]
            s1 = s.replace("\n", "")
            long_text = s1.replace("\r", "")
            f.write(long_text)
            f.write("\n")
            # 单独存储标签
            targets.append(content[2])
        f.close()
        #把标签列表转成pandas,存csv
        df=pd.DataFrame(targets)
        df.to_csv(self.targets_filename, header=False, index=False)

        print("data saved into txt file, finished! Using time:{}".format(arrow.now() - startTime))

    def save_csv(self,sql,csv_filename):
        startTime=arrow.now()
        datas=self.data_read(sql)

        df = pd.DataFrame(data=[], columns=["text", "tag"])
        for i in range(0,len(datas)):
            df.loc[i]=[datas[i][0]+" "+datas[i][1],datas[i][2]] #合并标题和文本

        df.to_csv(csv_filename,index=False)

        print("data saved into csv file, finished! Using time:{}".format(arrow.now()-startTime))



if __name__ == "__main__":

    url = "192.168.20.149"
    username = "root"
    password = "admin123!@#"
    db = "text_classification_samples"
    data=DataObj(url,username,password,db)

    positive_num,ratio=10,2
    sql="(SELECT title,content,information_type FROM samples_for_analysis where information_type=1 limit "+str(positive_num)+") " \
              "union all (SELECT title,content,information_type FROM samples_for_analysis where information_type=0 order by rand() limit " \
              +str(positive_num*ratio)+")"


    data.save_csv(sql,"raw_datas.csv")



