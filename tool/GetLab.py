import pandas as pd
data = pd.read_csv(r'C:\Users\WooL\Documents\GitHub\ACMS\newCSV\LabSet.csv')
Lab = data.drop_duplicates(subset=None, keep='first', inplace=False)
print(Lab)

SaveFile_Path =  r'C:\Users\WooL\Documents\GitHub\ACMS\newCSV'       #拼接後要儲存的檔案路徑
SaveFile_Name = r'LabFin.csv'
Lab.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False)