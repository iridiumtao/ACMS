import pandas as pd
import os
Folder_Path = r'C:\Users\WooL\Documents\GitHub\ACMS\csv'          #要拼接的資料夾及其完整路徑，注意不要包含中文
SaveFile_Path =  r'C:\Users\WooL\Documents\GitHub\ACMS\newCSV'       #拼接後要儲存的檔案路徑
SaveFile_Name = r'restSetBack.csv'              #合併後要儲存的檔名

os.chdir(Folder_Path)
file_list = os.listdir()
df = pd.read_csv(Folder_Path +'\\'+ file_list[0])
#df.insert(13,column=0,value=0)
df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False)
for i in range(1,len(file_list)):
    df = pd.read_csv(Folder_Path + '\\'+ file_list[i])
    #df.insert(13,column="class",value=i)
    df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')