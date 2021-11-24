import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.classifiers import Classifier
import weka.plot.graph as graph

jvm.start()

# load data
data = load_any_file("./data/iris.arff")
data.class_is_last()

# train classifier
cls = Classifier(classname="weka.classifiers.trees.J48",
                 options=["-C", "0.3"])
cls.build_classifier(data)
print(cls)

# display classifier graph
graph.plot_dot_graph(cls.graph)

# generate source
print(cls.to_source("IrisJ48"))

jvm.stop()
