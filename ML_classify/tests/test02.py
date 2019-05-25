import jieba
import jieba.posseg as pseg

content='一般不了芬达没有不用对比分红奖励奖金收益发帖子大家了解收益率15%三倍回报三级代理A区'

content1='2018年资金放大 封顶　1000—3万 5 倍   30000　公司发行资产包 10000 个，永不增发（一个资产包 1000 元）。放大后资产每天千分之二释放'

word_cut = pseg.cut(content1)
for word, flag in word_cut:
    print(word,flag)



