# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:37:37 2019

@author: Kunal
"""

import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import math as mt
from sklearn.decomposition import PCA


# Reading the input

file = open('german.data-numeric')
transaction_info = []
actual_classification = []
for line in file.readlines():
    line_data = []
    val = ''
    check = False
    for j in range(0,len(line)):
        if line[j] >= '0' and line[j] <= '9':
            val = val + line[j]
        elif val != '':
            line_data.append(int(val))
            val = ''
    last_value = line_data[len(line_data)-1]
    actual_classification.append(last_value)
    line_data = line_data[:-1]
    transaction_info.append(line_data)

#Calculating the mean and standard deviation for each attribute
    
attr_mean = []
attr_dev = []
for j in range(0,24):
    temp = []
    for i in range(0,len(transaction_info)):
        temp.append(transaction_info[i][j])
    attr_mean.append(st.mean(temp))
    attr_dev.append(st.stdev(temp))
    
#Normalizing the dataset
    
for j in range(0,24):
    for i in range(0,len(transaction_info)):
        transaction_info[i][j] = (transaction_info[i][j] - attr_mean[j])/attr_dev[j]
    
#Calculating the Euclidean Distance matrix
        

distance_matrix = []
for i in range(0,len(transaction_info)):
    temp = []
    for j in range(0,len(transaction_info)):
        sum = 0
        for k in range(0,24):
            sum = sum + (transaction_info[i][k]-transaction_info[j][k])*(transaction_info[i][k]-transaction_info[j][k])
        sum = mt.sqrt(sum)
        temp.append(sum)
    distance_matrix.append(temp)
    

#Calculating k-distance
    
k_min = 700
k_max = 702

lof = []

for i in range(0,len(transaction_info)):
    lof.append(0)

for k in range(k_min,k_max+1):
    k_distance = []
        
    for i in range(0,len(transaction_info)):
        temp = distance_matrix[i]
        temp.sort()
        k_distance.append(temp[k])
        
    #Calculating the Reachability Distance matrix
        
    reachability_distance = []
    
    for i in range(0,len(transaction_info)):
        temp = []
        for j in range(0,len(transaction_info)):
            val = max(k_distance[j],distance_matrix[i][j])
            temp.append(val)
        reachability_distance.append(temp)
            
    #Calculating LRD for each point
        
    lrd = []
    
    for i in range(0,len(transaction_info)):
        temp = []
        for j in range(0,len(transaction_info)):
            if j == i:
                continue
            temp.append((distance_matrix[i][j],j))
        temp.sort()
        sum = 0
        u = temp[k][0]
        sz = 0
        for j in range(0,999):
            if(u<temp[j][0]):
                break
            sum = sum + reachability_distance[i][temp[j][1]]
            sz = sz + 1
        sum = sum/sz
        ans = 1/sum
        lrd.append(ans)
    
    #Calculating LOF for each point
    
    for i in range(0,len(transaction_info)):
        temp = []
        for j in range(0,len(transaction_info)):
            if j == i:
                continue
            temp.append((distance_matrix[i][j],j))
        temp.sort()
        sum = 0
        u = temp[k][0]
        sz = 0
        for j in range(0,999):
            if(u<temp[j][0]):
                break
            sum = sum + lrd[temp[j][1]]/lrd[i]
            sz = sz + 1
        sum = sum/sz
        lof[i] = max(lof[i], sum)
    
# Classifying points
    
accuracy = 0
type = []
for i in range(0,len(transaction_info)):
    if(lof[i]>1):
        type.append(2)
    else:
        type.append(1)
    if(type[i] == actual_classification[i]):
        accuracy = accuracy + 1
accuracy = accuracy/(len(transaction_info))

#Creating the scatterplot
    
np_data = np.array(transaction_info)
pca = PCA(n_components = 2)
reduced_data = pca.fit_transform(np_data)
colors = []
for i in range(0,len(transaction_info)):
    if(type[i] == 2):
        colors.append('red')
    else:
        colors.append('blue')

X = []
Y = []

for i in range(0,len(transaction_info)):
    X.append(reduced_data[i][0])
    Y.append(reduced_data[i][1])

plt.title('Classification for points, k_min = ' + str(k_min) + ' , k_max = ' + str(k_max) + ', Accuracy = ' + str(accuracy))
plt.scatter(X,Y, color = colors)
plt.show()
