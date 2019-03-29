from dependencies.libraries import *
from config.constants import *
from util.generatingModule import *
from util.mlAlgorithms import *


def generate_predictions(saved_algorithm):
	predDf = sqlCtx.read.parquet(file_path)
	if get_cross_val_model:
		mdl = get_saved_model(saved_algorithm,saved_croos_val_model)
	else:
		mdl = get_saved_model(saved_algorithm,saved_model)
	
	pred = mdl.transform(predDf)
	evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction")
	print(evaluator.evaluate(pred))

	converter = IndexToString(inputCol="prediction", outputCol="predicted_class", labels=['Degraded', 'Non-Degraded'])
	converted = converter.transform(pred)
	columns = ['source_id', 'source_description', 'kpi_name', 'kpi_value', 'actual_class', 'site_id', 'metal', 'reading_date', 'kpi_id','predicted_class']
	converted.select( [c for c in converted.columns if c in columns]).show()

	
