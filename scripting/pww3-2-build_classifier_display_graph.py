import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.classifiers import Classifier
import weka.plot.graph as graph

jvm.start(packages=True)

data = converters.load_any_file("./data/iris.arff")
data.class_is_last()
cls = Classifier(classname="weka.classifiers.trees.J48", 
                 options=["-C", "0.3"])
cls.build_classifier(data)
print(cls)
graph.plot_dot_graph(cls.graph)

jvm.stop()

