import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.core.dataset import Instances
from weka.clusterers import Clusterer, ClusterEvaluation

jvm.start()

data = load_any_file("./data/iris.arff")
data_copy = Instances.copy_instances(data)
data_copy.class_is_last()

# remove class attribute
data.delete_last_attribute()

# build a clusterer and output model
clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
clusterer.build_clusterer(data)
print(clusterer)

# cluster data
print("--> Cluster instances")
for index, inst in enumerate(data):
    cl = clusterer.cluster_instance(inst)
    dist = clusterer.distribution_for_instance(inst)
    print(str(index + 1) + ": cluster=" + str(cl) + ", distribution=" + str(dist))

# classes to clusters
print("\n\n--> Classes to clusters")
evl = ClusterEvaluation()
evl.set_model(clusterer)
evl.test_model(data_copy)
print("Cluster results")
print(evl.cluster_results)
print("Classes to clusters")
print(evl.classes_to_clusters)

jvm.stop()
