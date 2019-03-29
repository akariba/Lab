from dependencies.libraries import *
from config.constants import *
from util.mlAlgorithms import *

def generate_model():
	#read train data from hdfs
	trainDf = sqlCtx.read.parquet(train_data)
	print("train data count is:",trainDf.count())

	#read test data from hdfs
	testDf = sqlCtx.read.parquet(test_data)
	print("test data count is:",testDf.count())

	alg = get_algorithm(algorithm_type)

	if is_crossvalidation:
		print("crossvalidation part")
		cv = cross_validation(alg)
		model = cv.fit(trainDf)
		if is_save_model:
			print("saving the cross-validation model")
			model.bestModel.write().overwrite().save(save_path+"/model/cross_validation/"+algorithm_type)
		else:
			print("Not saving cross-validation model")

	else:
		model = alg.fit(trainDf)
		if is_save_model:
			print("saving the model")
			model.write().overwrite().save(save_path + "/model/" + algorithm_type)

	if is_crossvalidation:
		predictions = model.bestModel.transform(testDf)
	else:
	    predictions = model.transform(testDf)

	#predictions.show()
	#evaluate the model
	evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction")
	print(evaluator.evaluate(predictions))

	if is_confusionmatrix:
	    get_confusionmatrix(predictions)

	print("generated the model successfully")   



