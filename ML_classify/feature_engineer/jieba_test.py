import jieba
jieba.load_userdict('myDict_20190326.txt')
import jieba.posseg as pseg

text="这个该死的传销名目有如下的奖金制度，它是一种五级三阶制制度，通过团队业绩计酬决定每位下线员工的收入，比如贡献奖之类的" \
     "传销人员通常喜欢拉动亲友进行虚假宣传并获取奖励无限层"

word_cut = pseg.cut(text)

for word, flag in word_cut:
    print(word)
