import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.core.dataset import Instances
from weka.clusterers import Clusterer

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
for index, inst in enumerate(data):
    cl = clusterer.cluster_instance(inst)
    dist = clusterer.distribution_for_instance(inst)
    act_inst = data_copy.get_instance(index)
    act = int(act_inst.get_value(data_copy.class_index))
    print(str(index + 1) + ": cluster=" + str(cl)
          + ", distribution=" + str(dist)
          + ", actual=" + str(act)
          + ", error=" + str(cl != act))

jvm.stop()
