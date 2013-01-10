import numpy
import scipy
import sys
import os
import random
#imports for running across cluster
from IPython.parallel import Client

sys.path.append("/home/charles/ml_implementations/PyTinyImage/")

import tinyimage

def argmin(clusters, sample):
	t = map(lambda c : numpy.linalg.norm(c - sample), clusters)
	return t.index(min(t))

def assignclusters(clusters, samples):
	assignments = []
	for s in samples:
		assignments.append(argmin(clusters, s))
	return zip(assignments, samples)	

def newclusters(cluster_count, cluster_dimension, samples):
 	clusters = map(lambda c : numpy.zeros(cluster_dimension), range(0, cluster_count))
	counters = map(lambda c : 0, range(0, cluster_count))
	for sample in samples:
		(index, data) = sample
		clusters[index] += data	
		counters[index] += 1
	return map(lambda o: o[0] / o[1], zip(clusters, counters))				
def clusterDiff(clusters_old, clusters_new):
	m = map(lambda o: numpy.linalg.norm(o[0] - o[1]), zip(clusters_old, clusters_new))
	return sum(m)

#now for the kmeans on our test images
print "Starting..."

keywords = ["cat", "dog", "house", "map", "toy"]
cluster_count = 5
max_pics = 2000
cluster_dimension = (32,32, 3)
means = []
convergence = 10
max_iterations = 5

#get the samples
samples = []

print "Loading images..."

tinyimage.openTinyImage()

for keyword in keywords:
	indexes = tinyimage.retrieveByTerm(keyword, max_pics)	
	temp = []
	for indx in indexes:
		samples.append(tinyimage.sliceToBin(indx).reshape(32,32,3, order="F"))

tinyimage.closeTinyImage()

print str(len(samples)) + " samples loaded..."
print "Seeding..."

#setup for cluster
cluster_client = Client()
print "cluster ids [for to verify parallel]"
print cluster_client.ids
cluster_dview = cluster_client[:]


#select random seeds from the samples
for i in range(0, cluster_count):
	r = random.randint(0, max_pics * cluster_count)
	means.append(numpy.empty(cluster_dimension))
	means[i][:] = samples[r]

cluster_dview.push(dict(means=means))

cluster_dview.scatter('samples', samples)

#perform k means
print "Clustering..."
a1 = assignclusters(means, samples)
nc1 = newclusters(cluster_count, cluster_dimension, a1)
d = clusterDiff(means, nc1)
iteration_count = 1

while d > convergence and iteration_count < max_iterations :
	print d
	means = nc1
	a1 = assignclusters(means, samples)
	nc1 = newclusters(cluster_count, cluster_dimension, a1)
	d = clusterDiff(means, nc1)
	iteration_count += 1

#finally perform our output
output_dir = "output"
for i in range(0, cluster_count):
	if not os.path.isdir(output_dir+"/"+str(i)):
		os.mkdir(output_dir+"/"+str(i))
	cluster_data_output = open(output_dir+"/"+str(i)+"/"+str(i)+".cluster", "wb")
	numpy.save(cluster_data_output, means[i])
	cluster_data_output.close()

for i in range(0, cluster_count * max_pics):
	(index, img_data) = a1[i]
	img = scipy.misc.toimage(img_data)
	img.save(output_dir+"/"+str(index)+"/"+str(i)+".png")				
	
