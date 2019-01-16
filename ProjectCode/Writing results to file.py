import csv
import xlwt
from tempfile import TemporaryFile
book = xlwt.Workbook()
for i in range(len(totalAccuracy)):
    sheet1 = book.add_sheet("sheets"+str(i+1))
    sheet1.write(0,1,"Accuracy")
    sheet1.write(0,2,"Positive Precision")
    sheet1.write(0,3,"Positive Recall")
    sheet1.write(0,4,"Negative Precision")
    sheet1.write(0,5,"Negative Recall")
    sheet1.write(0,6,"Positive F score")
    sheet1.write(0,7,"Negative F score")    
    for j in range(len(totalAccuracy[i])):
        sheet1.write(j+1,1,totalAccuracy[i][j][0])
        sheet1.write(j+1,2,totalAccuracy[i][j][1])
        sheet1.write(j+1,3,totalAccuracy[i][j][2])
        sheet1.write(j+1,4,totalAccuracy[i][j][3])
        sheet1.write(j+1,5,totalAccuracy[i][j][4])
        sheet1.write(j+1,6,totalAccuracy[i][j][5])
        sheet1.write(j+1,7,totalAccuracy[i][j][6])

name = "random.xls"
book.save(name)
book.save(TemporaryFile())
