import csv
import sys
import sklweka.jvm as jvm
import sklweka.packages as pkgs
from sklweka.dataset import load_arff
from sklweka.classifiers import WekaEstimator
from weka.classifiers import Classifier, FilteredClassifier, SingleClassifierEnhancer
from weka.filters import Filter, MultiFilter
from sklearn import metrics
import matplotlib.pyplot as plt

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
train_X, train_y, train_meta = load_arff("./data/NAnderson2020MendeleyMangoNIRData-cal.arff", class_index="2")
test_X, test_y, test_meta = load_arff("./data/NAnderson2020MendeleyMangoNIRData-val_ext.arff", class_index="2")

# configure classifier
# FilteredClassifier
# - MultiFilter
#   - Remove (SampleID)
#   - PLS (20 components)
# - LWL (250 neighbors)
#   - SMOreg
print("Configuring classifier")
lwl = SingleClassifierEnhancer(classname="weka.classifiers.lazy.LWL", options=["-K", "250"])
lwl.classifier = Classifier(classname="weka.classifiers.functions.SMOreg")

multi = MultiFilter()
multi.filters = [
    Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "1"]),
    Filter(classname="weka.filters.supervised.attribute.PLSFilter")
]

fc = FilteredClassifier()
fc.classifier = lwl
fc.filter = multi

# train classifier
print("Training classifier")
skl_fc = WekaEstimator(classifier=fc)
skl_fc.fit(train_X, train_y)

# evaluate classifier
print("Evaluating classifier")
test_predicted = skl_fc.predict(test_X)
mae = metrics.mean_absolute_error(test_y, test_predicted)
mse = metrics.mean_squared_error(test_y, test_predicted)
r2 = metrics.r2_score(test_y, test_predicted)
stats = [
    ["MAE", mae],
    ["MSE", mse],
    ["R2 score", r2],
]

# output results
# 1. statistics
with open("./output/statistics.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Statistic", "Value"])
    writer.writerows(stats)
# 2. errors plot
fig, ax = plt.subplots()
ax.scatter(test_y, test_predicted, edgecolors=(0, 0, 1))
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=3)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Classifier errors")
plt.savefig("./output/errors.png")
# 3. predictions
with open("./output/predictions.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Actual", "Predicted", "SampleID"])
    for i in range(len(test_y)):
        writer.writerow([test_y[i], test_predicted[i], test_X[i, 0]])

jvm.stop()
