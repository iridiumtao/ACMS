import csv

data=[]
with open(r'C:\Users\WooL\Documents\GitHub\ACMS\newCSV\restSet.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        print(row)

