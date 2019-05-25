import pandas as pd
import arrow
import numpy as np
from gensim.models import word2vec

df1=pd.DataFrame(data=[['483904823','就分厘卡机法国看来是','权游 雪诺 艾迪'],
                       ['439789793874','警方立刻撒旦解放拉萨扩大','冰火 龙母 异鬼'],
                       ['3920','附件是独立开发金克拉','一代宗师 受苦 受骗'],
                       ['4932p','负九点十六分就流口水的份','无风险 日方 日积月累'],
                       ['4832904','辣子草泥马','草泥马 去死吧']],
                 columns=['org_id','text','segment_text'])


df1.to_csv('datas/testDatas—1.csv',index=False)



