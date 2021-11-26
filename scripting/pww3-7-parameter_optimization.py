import sys
import weka.core.jvm as jvm
import weka.core.packages as pkgs
from weka.core.converters import load_any_file
from weka.classifiers import Classifier, MultiSearch
from weka.core.classes import ListParameter, MathParameter

jvm.start(packages=True)

# install multisearch package
if not pkgs.is_installed("multisearch"):
    print("Installing multisearch package")
    pkgs.install_package("multisearch")
    print("Please restart")
    jvm.stop()
    sys.exit()

# load dataset
train = load_any_file("./data/bolts.arff")
train.class_is_last()

# classifier: SMOreg using RBFKernel
# optimize:
# - gamma parameter of RBFKernel (via expression): 10^-3 to 10
# - SMOreg C parameter (list of values): -2 to 2
multi = MultiSearch(options=["-S", "1"])
multi.evaluation = "CC"
mparam = MathParameter()
mparam.prop = "kernel.gamma"
mparam.minimum = -3.0
mparam.maximum = 3.0
mparam.step = 1.0
mparam.base = 10.0
mparam.expression = "pow(BASE,I)"
lparam = ListParameter()
lparam.prop = "C"
lparam.values = ["-2.0", "-1.0", "0.0", "1.0", "2.0"]
multi.parameters = [mparam, lparam]
cls = Classifier(
    classname="weka.classifiers.functions.SMOreg",
    options=["-K", "weka.classifiers.functions.supportVector.RBFKernel"])
multi.classifier = cls
multi.build_classifier(train)
print("Model:\n" + str(multi))
print("\nBest setup:\n" + multi.best.to_commandline())

jvm.stop()
