import numpy as np
import statistics
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Function to calculate Euclidean distance
def distance(a,b):  
    dist = 0
    for j in range(0,24):
        dist = dist + ((data[a][j]-data[b][j])*(data[a][j]-data[b][j]))
        
    dist = math.sqrt(dist)
    
    return dist


file = open('german.data-numeric')

data = []

#Parsing the file
for line in file.readlines():
    line_data = []
    temp=''
    for i in range(0,len(line)):
        if(line[i]>='0' and line[i]<='9'):
            temp = temp + line[i]
        elif(temp!=''):
            line_data.append(int(temp))
            temp=''
    data.append(line_data)

attr_mean = []
attr_stdev = []

#Calculating mean and standard deviation of each attribute
for j in range(0,25):
    temp = []
    for i in range(0,1000):
        temp.append(data[i][j])
    attr_mean.append(statistics.mean(temp))
    attr_stdev.append(statistics.stdev(temp))
    
#Normalizing the data
for i in range(0,1000):
    for j in range(0,24):
        data[i][j] = (data[i][j]-attr_mean[j])/attr_stdev[j]
    
distance_matrix = []

#Calculating distance between every two points
for i in range(0,1000):
    temp = []
    for j in range(0,1000):
        temp.append(distance(i,j))
    distance_matrix.append(temp)

eps = 3.9
minpts = 23

classification = []

#DBSCAN algorithm
for i in range(0,1000):
    neighbours = 0
    for j in range(0,1000):
        if(distance_matrix[i][j]<=eps):
            neighbours = neighbours + 1
    if(neighbours>=minpts):
        classification.append(0)
    else:
        classification.append(2)
        
for i in range(0,1000):
    if(classification[i]==0):
        continue
    for j in range(0,1000):
        if(distance_matrix[i][j]<=eps and classification[j]==0):
            classification[i]=1
            
actual_classification=[]

for i in range(0,1000):
    actual_classification.append(data[i][24])
        
print(classification.count(0)+classification.count(1),actual_classification.count(1)) 
print(classification.count(2),actual_classification.count(2))        

temp_data = []

for i in range(0,1000):
    temp = []
    for j in range(0,24):
        temp.append(data[i][j])
    temp_data.append(temp)
    
np_data = np.array(temp_data)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(np_data)

colors = []

for i in range(0,1000):
    if(classification[i]==2):
        colors.append('red')
    elif classification[i]==0:
        colors.append('blue')
    else:
        colors.append('black')

X = []
Y = []

for i in range(0,1000):
    X.append(reduced_data[i][0])

for i in range(0,1000):
    Y.append(reduced_data[i][1])
    
plt.scatter(X,Y,color=colors)
plt.show()
    


