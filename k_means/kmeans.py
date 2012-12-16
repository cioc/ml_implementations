#i'll debug this later tonight or tommorrow morning

#basic kmeans(lloyd's) implementation (very simple)
import numpy
import scipy

def argmin(clusters, sample):
	return min(map(lambda c : numpy.linalg.norm(c - sample), clusters))

def newclusters(cluster_count, cluster_dimension, samples):
 	clusters = map(lambda c : numpy.zeros(cluster_dimension), range[0, cluster_count])
	counters = map(lambda c : 0, range[0, cluster_count])
	for sample in samples:
		(index, data) = sample
		clusters[index] += data	
		counters[index] += 1
	return map(lambda cl co: cl / co, zip(clusters, counters))				

def clusterDiff(clusters_old, clusters_new):
	return sum(lambda o n : numpy.linalg.norm(o - n), zip(clusters_old, clusters_new))

#now for the kmeans on our test images

cluster_dimension = (32,32, 3)

#TODO - START HERE WITH FINAL IMPLEMENTATION	
