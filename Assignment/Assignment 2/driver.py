import numpy as np
import k_means
import k_medians


dataArray = [[]]

# filepath = input("Enter data file path")
# filepath = "/home/kshitij/Downloads/sample"
filepath = "/home/kshitij/Downloads/assignment2fileformat_19856"
n = 0
with open(filepath, "r") as datafile:
    for line in datafile:
        n += 1
        let = line.split(",")
        let = let[:-1]

        temp = []
        for i in let:
            temp.append(int(i))
        dim = len(temp)
        dataArray.append(temp)

    dNo = n
    del dataArray[0]
    # print(dataArray)

clusteringMode = "1"
k = 3
t = 10

# clusteringMode = input("Enter 1 for k-means or 2 for k-medians")
# k = input("Enter number of clusters")
# k = int(k)
# t = input("Enter number of iterations to perform")
# t = int(t)

if clusteringMode == "1":
    # k_means(dataArray, k, dim, n, t)
    k_means.cluster(dataArray, k, dim, dNo, t)
elif clusteringMode == "2":
    # k_medians(dataArray, k, dim, n, t)
    k_medians.cluster(dataArray, k, dim, dNo, t)
else:
    print("Wrong Input")
