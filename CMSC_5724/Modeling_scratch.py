from DT_from_scratch import DecisionTree
import time

continuousFeatures = ['age', 'fnlwgt', 'capital-gain', 'education-num','capital-loss', 'hours-per-week']

continuousFeaturesIndices = [0,2,4,10,11,12]
features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']

discreteFeatures = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

def loadData(path):
    X = []
    y = []
    #count = 0
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue
            lineList = line.split(',')
            lineList = [val.strip() for val in lineList]
            if '?' in lineList:
                continue
            for feature in continuousFeatures:
                idx = features.index(feature)
                lineList[idx] = float(lineList[idx])
            X.append(lineList[:-2])
            y.append(lineList[-1])
    return X, y


X_train, y_train = loadData('originalData/adult.data')
X_test, y_test = loadData("originalData/adult.test")
clf = DecisionTree(min_samples_split=10,max_depth=10,n_feats=13,criterion="gini")

startTime = time.time()
clf.fit(X_train,y_train)
endTime = time.time()




y_pred = clf.predict(X_test)
fail = 0

for i in range(len(y_test)):
    if y_test[i] == "<=50K.":
        y_test[i] = "<=50K"
    else:
        y_test[i] = ">50K"
    if y_pred[i] is None:
        fail += 1

r = 0

print()
print("=============================================================================================================")
print("=========================================TRAINING ACCURACY AND LATENCY============================================")
print("=============================================================================================================")
print()

#print("[INFO] fail to predict num: ", fail)
for i in range(len(y_test)):
    #print(f"The {i}-th sample {X_test[i]}, the predicted result is {y_pred[i]}, the actual result is {y_test[i]}, the prediction is {y_test[i] == y_pred[i]}")
    if y_test[i] == y_pred[i]:
        r += 1

print("accuracy: ", r / (len(y_pred)))
print(f"training latency:{endTime - startTime}")

print()
print("=============================================================================================================")
print("=========================================VISUALIZATION OF THE DECISION TREE========================================= =")
print("=============================================================================================================")
print()

def printTree(node, level=0):
    if node != None:
        printTree(node.left, level + 1)
        if node.is_leaf() == True:
            print(' ' * 20 * level + '->', node.value, f"level: {level}")
        else:
            if node.feature not in continuousFeaturesIndices:
                print(' ' * 20 * level + '->', f"{features[node.feature]} == {node.threshold}", f"level: {level}")
            else:
                print(' ' * 20 * level + '->', f"{features[node.feature]} <= {node.threshold}", f"level: {level}")
        printTree(node.right, level + 1)
printTree(clf.root)

print()
print("=============================================================================================================")
print("=========================================THE RESULT FOR EACH SAMPLE==========================================")
print("=============================================================================================================")
print()
for i in range(len(y_test)):
    print(f"The {i+1}-th sample {X_test[i]}, the predicted result is {y_pred[i]}, the actual result is {y_test[i]}, the prediction is {y_test[i] == y_pred[i]}")