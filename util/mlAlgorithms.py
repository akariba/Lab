from dependencies.libraries import *
from config.constants import *

def get_algorithm(algorithm):
    if algorithm == "randomforest":
        alg = random_forest()
        print(type(alg))
        return alg

    elif algorithm == "decisiontree":
        alg = decision_tree()
        print(type(alg))
        return alg
    elif algorithm == "gbtclassifier":
        alg = gbt_classifier()
        print(type(alg))
        return alg  
    elif algorithm == "logregression":
         alg = log_regression()
         print(type(alg))
         return alg

    else:
        print("No algorithms selected")

def random_forest():
	rf = RandomForestClassifier(numTrees=randomforest_tree_no, maxDepth=randomforest_depth, impurity="entropy",labelCol="label",featuresCol="features")
	return rf

def decision_tree():
	dt = DecisionTreeClassifier(labelCol="label",featuresCol="features",impurity="entropy")
	return dt

def gbt_classifier():
    gbt = GBTClassifier(featuresCol='features', labelCol='label')
    return gbt	

def log_regression():
    logr = LogisticRegression(featuresCol='features', labelCol='label')
    return logr

def cross_validation(alg):
    paramGrid = ParamGridBuilder().addGrid(alg.maxDepth, [5,10,15]).build()
    crossval = CrossValidator(estimator=alg,estimatorParamMaps=paramGrid,evaluator=MulticlassClassificationEvaluator(),numFolds=cross_val_num_folds)
    return crossval  

def get_confusionmatrix(predictions):
    print("confusion_matrix is:")
    predictionsAndLabels = predictions.select("prediction", "label").rdd
    metrics = MulticlassMetrics(predictionsAndLabels)
    confusion_matrix = metrics.confusionMatrix().toArray()
    print(confusion_matrix)

def get_saved_model(saved_alg,model_path):
    if saved_alg == "randomforest":
        saved_mdl = RandomForestClassificationModel.load(model_path+saved_alg)
        return saved_mdl
    elif saved_alg == "decisiontree":
        print(model_path+saved_alg)
        saved_mdl = DecisionTreeClassificationModel.load(model_path+saved_alg)
        return saved_mdl
    elif saved_alg == "gbtclassifier":
        saved_mdl = GBTClassifierModel.load(model_path+saved_alg)
        return saved_mdl
    elif saved_alg == "logregression":
        saved_mdl = LogisticRegressionModel.load(model_path+saved_alg)
        return saved_mdl
    else:
        print("No proper algorithm selected")