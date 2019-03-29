root_path="hdfs:/ml/csdp"
train_data = root_path+"/extracted_test_train_data/traindata/"
test_data = root_path+"/extracted_test_train_data/testdata/"
save_path = root_path

#generating model constants 
is_build_model = True
algorithm_type = "gbtclassifier"
is_crossvalidation = True
is_save_model = True
cross_val_num_folds = 10
is_confusionmatrix = False
randomforest_depth = 15
randomforest_tree_no = 50
randomforest_impurity = "entropy"

#generating prediction constants
get_cross_val_model = False
saved_algorithm = "randomforest"
saved_model = root_path+"/model/"
saved_croos_val_model = root_path+"/model/cross_validation/"
file_path = root_path+"/extracted_test_train_data/testdata/"
