# to run this program, use the instruction here in the console using python3,
python Modeling_scratch.py > log.txt

# main python files
# Modeling_scratch.py is the main function you need to run
# DT_from_scratch.py is the realization of the decision tree structure and optimization

# The output file log.txt includes:
#1. a visualized dicision tree for the training set
#2. a detailed list of all the records in the evaluation set, for each record its attributes and whether it has been classified successfully. 

# to changed the parameter, you should alter the this line from Modeling_scratch, the n_feats is fixed, and the criterion only contains "gini" and "entropy"
clf = DecisionTree(min_samples_split=10,max_depth=10,n_feats=13,criterion="gini")

# the history_procedure folder contains the source codes of the previous editions and should not be used in the current procedure. They are only used for 
# tracking the previous editions.

# CMSC5724 Project Report is the report we wrote which describing the decision tree and the details
