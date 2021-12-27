import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DT_numpy_enhanced import DecisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

arr = np.loadtxt('../originalData/adult.data',delimiter=', ',dtype=np.str_,encoding='utf-8')


# deleting the missing values and last column native country
arr = np.delete(arr, np.unique(np.where(arr == "?")[0]),axis=0)

y = arr[:,-1]
X = arr[:,:-2]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf = DecisionTree(min_samples_split=10,max_depth=10,criterion="gini")
import time
startTime = time.time()
clf.fit(X_train,y_train)
endTime = time.time()
print(f"training latency:{endTime - startTime}")


y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)