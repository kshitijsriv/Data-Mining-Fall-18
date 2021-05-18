import numpy as np
import general as gen

def cluster(dataArray, k, dim, dNo, t):
    reps = gen.initializeRandom(dataArray, k, dim, dNo)

    for itr in range(t):
        n = []
        clusters = gen.clustering_k_medians(dataArray, k, reps, dNo)
        for i in range(k):
            n.append(gen.findmedianofcluster(clusters[i]))

        if ((np.array(n) - np.array(reps)).any()) <= 0.00000001:
            # print(itr)
            break

        reps = n
    # print(n)
    for i in range(k):
        print("Cluster #" + str(i))
        print("--------------------------")
        print(clusters[i])
        print("--------------------------")
    return None
