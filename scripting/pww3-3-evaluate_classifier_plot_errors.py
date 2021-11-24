import csv
import sys
import weka.core.jvm as jvm
import weka.core.packages as pkgs
from weka.core.converters import load_any_file
from weka.classifiers import Evaluation, Classifier, FilteredClassifier, SingleClassifierEnhancer
from weka.filters import Filter, MultiFilter
import weka.plot.classifiers as plcls

jvm.start(packages=True, max_heap_size="512m")

# install PLS package
if not pkgs.is_installed("partialLeastSquares"):
    print("Installing PLS package")
    pkgs.install_package("partialLeastSquares")
    print("Please restart")
    jvm.stop()
    sys.exit()

# load data
print("Loading data")
train = load_any_file("./data/NAnderson2020MendeleyMangoNIRData-cal.arff")
dm_index = train.attribute_by_name("DM").index
train.class_index = dm_index
test = load_any_file("./data/NAnderson2020MendeleyMangoNIRData-val_ext.arff")
test.class_index = dm_index

msg = train.equal_headers(test)
if msg is not None:
    print("Incompatible datasets!")
    jvm.stop()
    sys.exit(1)

# configure classifier
# FilteredClassifier
# - MultiFilter
#   - Remove (1-11)
#   - PLS (20 components)
# - LWL (250 neighbors)
#   - SMOreg
print("Configuring classifier")
lwl = SingleClassifierEnhancer(classname="weka.classifiers.lazy.LWL", options=["-K", "250"])
lwl.classifier = Classifier(classname="weka.classifiers.functions.SMOreg")

multi = MultiFilter()
multi.filters = [
    Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "12-last", "-V"]),
    Filter(classname="weka.filters.supervised.attribute.PLSFilter")
]

fc = FilteredClassifier()
fc.classifier = lwl
fc.filter = multi

# train classifier
print("Training classifier")
fc.build_classifier(train)

# evaluate classifier
print("Evaluating classifier")
evl = Evaluation(train)
evl.test_model(fc, test)

# output results
# 1. summary
with open("./output/summary.txt", "w") as f:
    f.write(evl.summary())
# 2. errors plot
plcls.plot_classifier_errors(evl.predictions, wait=False, outfile="./output/errors.png")
# 3. predictions
sid_index = train.attribute_by_name("SampleID").index
preds = []
for i, pred in enumerate(evl.predictions):
    preds.append([pred.actual, pred.predicted, test.get_instance(i).get_string_value(sid_index)])
with open("./output/predictions.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Actual", "Predicted", "SampleID"])
    writer.writerows(preds)

jvm.stop()
