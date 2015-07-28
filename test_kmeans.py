#################################################  
# kmeans: k-means cluster  
# Author :  
# Date   :   
# HomePage :  
# Email  :  
#################################################  
  
from numpy import *  
import time  
import matplotlib.pyplot as plt 
import KMeans
   
## step 1: load data  
print ("step 1: load data..." ) 
dataSet = []   #列表，用来表示，列表中的每个元素也是一个二维的列表；这个二维列表就是一个样本，样本中包含有我们的属性值和类别号。
#与我们所熟悉的矩阵类似，最终我们将获得N*2的矩阵，每行元素构成了我们的训练样本的属性值和类别号
fileIn = open("D:/xuepython/testSet.txt")  #是正斜杠
for line in fileIn.readlines(): 
	temp=[]
	lineArr = line.strip().split('\t')  #line.strip()把末尾的'\n'去掉
	temp.append(float(lineArr[0]))
	temp.append(float(lineArr[1]))
	dataSet.append(temp)
    #dataSet.append([float(lineArr[0]), float(lineArr[1])])  
fileIn.close()  
## step 2: clustering...  
print ("step 2: clustering..."  )
dataSet = mat(dataSet)  #mat()函数是Numpy中的库函数，将数组转化为矩阵
k = 4  
centroids, clusterAssment = KMeans.kmeans(dataSet, k)  #调用KMeans文件中定义的kmeans方法。
  
## step 3: show the result  
print ("step 3: show the result..."  )
KMeans.showCluster(dataSet, k, centroids, clusterAssment)