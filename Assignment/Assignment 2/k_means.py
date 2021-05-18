import numpy as np
import general as gen


def cluster(dataArray, k, dim, dNo, t):
    # reps = gen.initializeRandom(dataArray, k, dim, dNo)
    # print(reps)
    # reps = np.array([[1], [2], [3]])
    reps = np.array([[1], [11], [28]])
    print(reps)
    for itr in range(t):
        n = []
        # print(dataArray)
        clusters = gen.clustering_k_means(dataArray, k, reps, dNo)
        # print(clusters)
        for i in range(k):
            n.append(gen.findmeanofcluster(clusters[i]))

        if ((np.array(n) - np.array(reps)).any()) <= 0.00000001:
            print("Number of iterations = " + str(itr))
            break
        reps = n
    # print(n)
    for i in range(k):
        print("Cluster #" + str(i))
        print("--------------------------")
        print(clusters[i])
        print("--------------------------")
    return None
