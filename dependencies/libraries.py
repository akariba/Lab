import sys
import json
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorIndexer, IndexToString, StringIndexerModel,VectorAssembler,OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel,GBTClassifier,DecisionTreeClassificationModel,GBTClassificationModel
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.sql import HiveContext
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext()
sqlCtx = HiveContext(sc)
SparkSession.builder.getOrCreate()
SparkSession.builder.appName("ML Algorithms").getOrCreate().sparkContext.setLogLevel("ERROR")