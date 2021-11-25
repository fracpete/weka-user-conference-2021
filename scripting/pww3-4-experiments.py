import tempfile
import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.classifiers import Classifier
from weka.experiments import SimpleCrossValidationExperiment, Tester, ResultMatrix

jvm.start()

datasets = ["./data/iris.arff", "./data/anneal.arff"]
classifiers = [Classifier("weka.classifiers.rules.ZeroR"), Classifier("weka.classifiers.trees.J48")]
outfile = tempfile.gettempdir() + "/results-cv.arff"
exp = SimpleCrossValidationExperiment(
    classification=True,
    runs=10,
    folds=10,
    datasets=datasets,
    classifiers=classifiers,
    result=outfile)
exp.setup()
exp.run()

# evaluate
data = load_any_file(outfile)
matrix = ResultMatrix("weka.experiment.ResultMatrixPlainText")
tester = Tester("weka.experiment.PairedCorrectedTTester")
tester.resultmatrix = matrix
comparison_col = data.attribute_by_name("Percent_correct").index
tester.instances = data
print(tester.header(comparison_col))
print(tester.multi_resultset_full(0, comparison_col))

jvm.stop()
