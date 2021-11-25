import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection

jvm.start()

data = load_any_file("./data/anneal.arff")
data.class_is_last()

# perform attribute selection
print("\n\n--> Attribute selection")
search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluation)
attsel.select_attributes(data)
print("# attributes: " + str(attsel.number_attributes_selected))
print("attributes (as numpy array): " + str(attsel.selected_attributes))
print("attributes (as list): " + str(list(attsel.selected_attributes)))
print("result string:\n" + attsel.results_string)

# perform ranking
print("\n\n--> Attribute ranking (2-fold CV)")
search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
evaluation = ASEvaluation("weka.attributeSelection.InfoGainAttributeEval")
attsel = AttributeSelection()
attsel.ranking(True)
attsel.folds(2)
attsel.crossvalidation(True)
attsel.seed(42)
attsel.search(search)
attsel.evaluator(evaluation)
attsel.select_attributes(data)
print("ranked attributes:\n" + str(attsel.ranked_attributes))
print("result string:\n" + attsel.results_string)

# transform data
print("\n\n--> Transform data")
search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
evaluation = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents", options=[])
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluation)
attsel.select_attributes(data)
print("transformed header:\n" + str(evaluation.transformed_header()))
#print("\ntransformed data:\n" + str(evaluation.transformed_data(data)))
print("\nconvert instance:\n" + str(evaluation.convert_instance(data.get_instance(0))))

jvm.stop()
