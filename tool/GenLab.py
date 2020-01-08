import pandas as pd
import numpy as np
data = pd.read_csv(r'C:\Users\WooL\Documents\GitHub\ACMS\newCSV\restSet.csv') 
df = pd.DataFrame(data,columns=['0'])
df1 = pd.DataFrame(data,columns=['1'])
df2 = pd.DataFrame(data,columns=['2'])
df3 = pd.DataFrame(data,columns=['3'])
df4 = pd.DataFrame(data,columns=['4'])
df5 = pd.DataFrame(data,columns=['5'])
df6 = pd.DataFrame(data,columns=['6'])
df7 = pd.DataFrame(data,columns=['7'])
df8 = pd.DataFrame(data,columns=['8'])
df9 = pd.DataFrame(data,columns=['9'])
connect=[df1,df2,df3,df4,df5,df6,df7,df8,df9]
dfFin = df.append(connect, ignore_index=False)
#dfFin = [i for i in dfFin if i !='']
#dfFin.drop_duplicates(subset=None, keep='first', inplace=False)
print(dfFin)