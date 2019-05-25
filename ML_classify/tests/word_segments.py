import pkuseg

import jieba.posseg as pseg

test_str="上海微阵科技有限公司引发了业内有关白酒与微商嫁接的遐想。" \
         "这种五级三阶制的奖金制度是典型的传销特征，还有每月返还的分红开发，团队业绩按照人头收益来计算。"

seg = pkuseg.pkuseg(postag=True,user_dict="my_dict_20190314.txt")  # 开启词性标注功能
text = seg.cut(test_str) # 进行分词和词性标注

print("using pkuseg:")
print(text)

print("using jieba:")
wordList=[]
word_cut = pseg.cut(test_str)
for word,flag in word_cut:
    wordList.append((word,flag))
print(wordList)

