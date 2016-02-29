import csv
with open('tst.csv','rb') as csvf:
    csvf.seek(3)
    f = csv.reader(csvf)
    for line in f:
        print line
