import numpy as np


def initializeRandom(dataArray, number_of_clusters, dim, dNo):
    # return k_initial_seeds
    initialseed = []
    for i in range(number_of_clusters):
        initialseed.append([])
        mx = np.max(dataArray, axis=0)
        mn = np.min(dataArray, axis=0)

        for j in range(len(mx)):
            initialseed[i].append(np.random.randint(mn[j], mx[j] + 1))
    reps = np.array(initialseed)
    return reps


def euclideanDistance(rep, data):
    # print(type(rep))
    # print(type(data))
    # return np.sum(np.sqrt(np.abs(rep**2-data**2)))
    # print(np.linalg.norm(rep - data))
    return np.linalg.norm(rep - data)



def manhattanDistance(rep, data):
    return np.sum(np.abs(rep - data), axis=0)


def clustering_k_means(datapointarray, k, reps, dNo):

    clusters = []

    for i in range(dNo):
        data = datapointarray[i]
        dist = []
        for j in range(int(k)):
            clusters.append([])
            # print(reps[j])
            dis = euclideanDistance(reps[j], data)
            dist.append(dis)
        # print("distance = ", dist)
        cNo = dist.index(min(dist))
        # print("Cluster assigned = ", cNo)
        clusters[cNo].append(data)

    return clusters

    # calculate distance from representative [call distance function]
    # assign to cluster
    # find cluster mean [call mean function]
    # set new cluster representative


def clustering_k_medians(datapointarray, k, reps, dNo):

    clusters = []

    for i in range(dNo):
        data = datapointarray[i]
        dist = []
        for j in range(int(k)):
            clusters.append([])
            dis = manhattanDistance(reps[j], data)
            dist.append(dis)

        cNo = dist.index(min(dist))
        clusters[cNo].append(data)

    return clusters


def findmeanofcluster(cluster):
    # cluster = np.delete(cluster, [0], axis=0)
    return np.mean(cluster, axis=0)


def findmedianofcluster(cluster):
    # return median
    return np.median(cluster, axis=0)

