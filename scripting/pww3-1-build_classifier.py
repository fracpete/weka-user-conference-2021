import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.classifiers import Classifier

jvm.start()

# load data
data = load_any_file("./data/iris.arff")
data.class_is_last()

# train classifier
cls = Classifier(classname="weka.classifiers.trees.J48",
                 options=["-C", "0.3"])
cls.build_classifier(data)
print(cls)

jvm.stop()
