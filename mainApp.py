import os
import sys

pwd = os.getcwd()
print(pwd)
#sys.path.insert(0,pwd)

from dependencies.libraries import *
from config.constants import *
from util.mlAlgorithms import *
from util.generatingModule import *
from util.generatingPredictions import *


if is_build_model:
    generate_model()
else:
    generate_predictions(saved_algorithm)    

print("completed successfully")
sc.stop()       