import sklweka.jvm as jvm
from sklweka.dataset import load_arff
from sklweka.classifiers import WekaEstimator
from sklearn.model_selection import cross_val_score

jvm.start(packages=True)

# load data
X, y, meta = load_arff("./data/bolts.arff", class_index="last")

# train classifier
lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
scores = cross_val_score(lr, X, y, cv=10,
                         scoring='neg_root_mean_squared_error')
print("Cross-validating LR on bolts (negRMSE)\n", scores)

jvm.stop()
