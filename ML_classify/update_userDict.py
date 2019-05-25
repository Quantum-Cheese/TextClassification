import pymysql


def update_keyWords():
    conn = pymysql.connect(host="192.168.20.42", user="root", passwd="123456", db="les_rulelib",
                           charset='utf8')
    curs = conn.cursor()
    curs.execute("select term from les_cx_features")
    result = curs.fetchall()
    conn.close()

    lst=[r[0]+" "+"2000\n" for r in result]
    long_text=('').join(lst)

    with open("datas/myDict.txt",'w',encoding='utf-8') as f:
        f.write(long_text)


if __name__ == "__main__":
    update_keyWords()

