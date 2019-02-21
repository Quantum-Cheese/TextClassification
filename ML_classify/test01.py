import numpy as np
d=np.array([[0],[0],[1],[1]])
f=d.ravel()
# print(d)
# print(f)

import pandas as pd

df1=pd.read_csv("features.csv",header=None)
df2=pd.read_csv("targets.csv",header=None)

print(df1)
print(df2)