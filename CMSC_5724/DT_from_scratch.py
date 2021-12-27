# =====================================================================
# Part I import useful packages
# =====================================================================

import random
from collections import Counter
import math

# =====================================================================
# Part II predefine parameters
# =====================================================================

# the list of continuous features
continuousFeatures = ['age', 'fnlwgt', 'capital-gain', 'education-num','capital-loss', 'hours-per-week']
# the indices of continuous feature indices
continuousFeaturesIndices = [0,2,4,10,11,12]


# =====================================================================
# Part III predefine functions
# =====================================================================

# get the table with two columns using tuples to include them
def getValuesPairs(matrix, col, labels):
    dataPairs = []
    data = set()
    for idx, l in enumerate(matrix):
        dataPairs.append((l[col], labels[idx]))
        data.add(l[col])
    return dataPairs, sorted(list(data))

# calculate the entropy
def entropy(y):
    counter = Counter(y)
    ps = [counter.most_common()[i][1]/len(y) for i in range(len(counter.most_common()))]
    return -sum([p * math.log2(p) for p in ps])

# calculate the gini index of one column
def gini(y):
    counter = Counter(y)
    ps = [counter.most_common()[i][1]/len(y) for i in range(len(counter.most_common()))]
    return 1-sum([p**2 for p in ps])

# get shuffle the feature list
def choice(n_features):
    ls_tmp = [i for i in range(n_features)]
    random.shuffle(ls_tmp)
    return ls_tmp

'''
get values like
[[x],[y],...,[n]]
, the x,y to n rows for the whole dataset. THe values in it can be the length of the specified columns
'''

def getColumns(ls,start,end):
    result = [i[start:end+1] for i in ls]
    return result

# get one column in the dataframe
def getOneColumn(ls,col):
    result = [i[col] for i in ls]
    return result

# get
def getValueByRow(dataLIst, rowList):
    data = []
    for row in rowList:
        data.append(dataLIst[row])
    return data

def getRows(ls,idxs):
    data = []
    for row in idxs:
        data.append(ls[row])
    return data


def getLenOfUniqueRows(X):
    return len(set([tuple(i) for i in X]))

class Node:
    # need to store the best split feature, and threshold. also the left and right child tree,
    # at leaf node we need to store the most common labels

    def __init__(self, feature=None, threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


# n_feats -> subset of number of features
class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=10,n_feats=None,criterion="gini"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_feats = n_feats
        # later need to know the root
        self.root = None

    def fit(self,X,y):
        # training a DT
        #self.n_feats = 13
        self.root = self._grow_tree(X,y)


    def _grow_tree(self,X,y,depth = 0):
        n_samples, n_features = len(X),len(X[0])
        n_labels = len(set(y))

        # stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
            or getLenOfUniqueRows(X) <= 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # select random features, the returned value should be a list
        feat_idxs = choice(n_features)
        # greedy search to find the best split
        best_feat, best_thresh = self._best_criteria(X,y,feat_idxs)

        left_idxs, right_idxs = self._split(best_feat, getOneColumn(X,best_feat), best_thresh)
        left = self._grow_tree(getRows(X,left_idxs), getRows(y,left_idxs), depth+1)
        right = self._grow_tree(getRows(X,right_idxs), getRows(y,right_idxs), depth+1)

        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self,X,y,feat_idxs):
        split_idx, split_thresh = None, None
        if self.criterion == "entropy":
            best_gain = -1
            for feat_idx in feat_idxs:
            # every time we try one feature
                # get the selected column as the split
                X_column = getOneColumn(X,feat_idx)
                if feat_idx in continuousFeaturesIndices:
                    parent_entropy = entropy(y)
                    X_table = getValuesPairs(X, feat_idx, y)[0]

                    ChildEntropy, threshold = self._optimized_entropy_table_continuous(X_table)
                    gain = parent_entropy - ChildEntropy
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = threshold
                        # print(feat_idx,threshold)
                else:
                    thresholds = list(set(X_column)) # get all unique values
                    for threshold in thresholds:
                        gain = self._information_gain(feat_idx,y,X_column, threshold)
                        if gain > best_gain:
                            best_gain = gain
                            split_idx = feat_idx
                            split_thresh = threshold

        elif self.criterion == "gini":
            smallest_gini_split = 1
            # the tmp gini values
            gini_split = 1
            # the threshold is the minimum value's index
            threshold = None

            # two for loops to traverse all features and all values
            for feat_idx in feat_idxs:

                X_column = getOneColumn(X,feat_idx)
                if feat_idx in continuousFeaturesIndices:

                    # for continuous value, reduce the time complexity
                    # TODO create the big table, the first column is the attr, second is the label, 3-> C1y, 4->C1n,5->C2y,6->C2n
                    # TODO this should be changed after replaces all numpy functions

                    X_table = getValuesPairs(X,feat_idx,y)[0]

                    gini_split, threshold = self._optimized_gini_table_continuous(X_table)
                    if gini_split < smallest_gini_split:
                        smallest_gini_split = gini_split
                        split_idx = feat_idx
                        split_thresh = threshold
                        #print(feat_idx,threshold)

                else:
                    # for the discrete values
                    thresholds = list(set(X_column)) # get all unique values
                    for threshold in thresholds:
                        # in gini index the best gain is the smallest gini
                        gini_split = self._gini_split(feat_idx,y,X_column, threshold)
                        if gini_split < smallest_gini_split:
                            smallest_gini_split = gini_split
                            split_idx = feat_idx
                            split_thresh = threshold
                #print(split_idx,smallest_gini_split,split_thresh)
            #print("smallest_gini_split:",smallest_gini_split,split_idx,split_thresh)
        return split_idx, split_thresh

    def _optimized_gini_table_continuous(self,X_table):
        # sort the table by its attris
        X_table = sorted(X_table, key=lambda x: x[0])
        total_len = len(X_table)
        # the count is used to record the duplicated times,very important
        count =[]
        idx = 0

        while idx < total_len:
            c1n_group=0
            c1y_group=0

            duplicated_times = 1
            if X_table[idx][1] == X_table[0][1]:
                #c1y += 1
                c1y_group+=1
            else:
                #c1n += 1
                c1n_group+=1
            # keep looping if duplicated, and keep adding
            while idx+1 < total_len and X_table[idx][0] == X_table[idx+1][0]:
                idx += 1
                duplicated_times+=1
                if X_table[idx][1] == X_table[0][1]:
                    #c1y += 1
                    c1y_group+=1
                else:
                    #c1n += 1
                    c1n_group+=1

            count.append((X_table[idx][0],duplicated_times,c1y_group,c1n_group))
            #C1Y.extend(duplicated_times*[c1y])
            #C1N.extend(duplicated_times*[c1n])
            idx += 1
        if len(count) == 1:
            return 1, 0
        #print(f"count:{count}")
        # Part generate the final results according to the "count"

        # this for loop needs to cal„culate four lists, each contains the number for split set 1 and 2 respectively
        c1y=0
        c1n=0
        c2y=0
        c2n=0

        C1Y=[]
        C1N=[]
        # the first two values for set two should be set to 0, since our strategy is "<="
        C2Y=[]
        C2N=[]

        for idx in range(len(count)-1):
            reversed_idx = len(count) - idx -1
            c1y += count[idx][2]
            C1Y.append(c1y)

            c1n += count[idx][3]
            C1N.append(c1n)

            # calculate the reversed items
            c2y += count[reversed_idx][2]
            C2Y.append(c2y)

            c2n += count[reversed_idx][3]
            C2N.append(c2n)

        C2N.reverse()
        C2Y.reverse()
        # correct ALL lists:


        SIZE_LIST_1  = [x + y for x, y in zip(C1Y, C1N)]

        SIZE_LIST_2 = [x + y for x, y in zip(C2Y, C2N)]

        # calculation of gini values
        gini_list = self._gini_calculate_optimized(C1Y, C1N, C2Y, C2N, SIZE_LIST_1, SIZE_LIST_2,total_len)
        # return the minimun value's index as the threshold

        index = min(range(len(gini_list)), key=gini_list.__getitem__)
        return min(gini_list), count[index][0]

    def _gini_calculate_optimized(self,C1Y, C1N, C2Y, C2N, SIZE_LIST_1, SIZE_LIST_2,total_len):
        gini_list =[]
        for c1y, c1n, c2y, c2n, size1, size2 in zip(C1Y, C1N, C2Y, C2N, SIZE_LIST_1, SIZE_LIST_2):
            weight1 = (size1/total_len)
            weight2 = (size2/total_len)
            gini1 = (1 - pow(c1y/size1,2) - pow(c1n/size1,2))
            gini2 = (1 - pow(c2y/size2,2) - pow(c2n/size2,2))
            gini_list.append(weight1*gini1+weight2*gini2)
        return gini_list

    def _information_gain(self,feat_idx,y,X_column, split_thresh):
        # parent Entropy
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(feat_idx,X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted avg child Entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(getRows(y,left_idxs)), entropy(getRows(y,right_idxs))
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # return ig
        ig = parent_entropy - child_entropy
        return ig

    def _optimized_entropy_table_continuous(self,X_table):
        # sort the table by its attris
        X_table = sorted(X_table, key=lambda x: x[0])
        total_len = len(X_table)
        # the count is used to record the duplicated times,very important
        count =[]
        idx = 0

        while idx < total_len:
            c1n_group=0
            c1y_group=0

            duplicated_times = 1
            if X_table[idx][1] == X_table[0][1]:
                #c1y += 1
                c1y_group+=1
            else:
                #c1n += 1
                c1n_group+=1
            # keep looping if duplicated, and keep adding
            while idx+1 < total_len and X_table[idx][0] == X_table[idx+1][0]:
                idx += 1
                duplicated_times+=1
                if X_table[idx][1] == X_table[0][1]:
                    #c1y += 1
                    c1y_group+=1
                else:
                    #c1n += 1
                    c1n_group+=1

            count.append((X_table[idx][0],duplicated_times,c1y_group,c1n_group))
            #C1Y.extend(duplicated_times*[c1y])
            #C1N.extend(duplicated_times*[c1n])
            idx += 1
        if len(count) == 1:
            return 1, 0
        #print(f"count:{count}")
        # Part generate the final results according to the "count"

        # this for loop needs to cal„culate four lists, each contains the number for split set 1 and 2 respectively
        c1y=0
        c1n=0
        c2y=0
        c2n=0

        C1Y=[]
        C1N=[]
        # the first two values for set two should be set to 0, since our strategy is "<="
        C2Y=[]
        C2N=[]

        for idx in range(len(count)-1):
            reversed_idx = len(count) - idx -1
            c1y += count[idx][2]
            C1Y.append(c1y)

            c1n += count[idx][3]
            C1N.append(c1n)

            # calculate the reversed items
            c2y += count[reversed_idx][2]
            C2Y.append(c2y)

            c2n += count[reversed_idx][3]
            C2N.append(c2n)

        C2N.reverse()
        C2Y.reverse()
        # correct ALL lists:


        SIZE_LIST_1  = [x + y for x, y in zip(C1Y, C1N)]

        SIZE_LIST_2 = [x + y for x, y in zip(C2Y, C2N)]

        # calculation of gini values
        entropy_list = self._entropy_calculate_optimized(C1Y, C1N, C2Y, C2N, SIZE_LIST_1, SIZE_LIST_2,total_len)
        # return the minimun value's index as the threshold

        index = min(range(len(entropy_list)), key=entropy_list.__getitem__)
        return min(entropy_list), count[index][0]


    def _entropy_calculate_optimized(self,C1Y, C1N, C2Y, C2N, SIZE_LIST_1, SIZE_LIST_2,total_len):
        entropy_list =[]
        for c1y, c1n, c2y, c2n, size1, size2 in zip(C1Y, C1N, C2Y, C2N, SIZE_LIST_1, SIZE_LIST_2):
            weight1 = (size1/total_len)
            weight2 = (size2/total_len)
            if c1y/size1 == 0 or c1n/size1 == 0:
                entropy1 = 0
            else:
                entropy1 =  -(c1y/size1)*math.log2(c1y/size1) - (c1n/size1)*math.log2(c1n/size1)
            if c2y/size2 == 0 or c2n/size2 == 0:
                entropy2 = 0
            else:
                entropy2 =  -(c2y/size2)*math.log2(c2y/size2) - (c2n/size2)*math.log2(c2n/size2)
            entropy_list.append(weight1*entropy1+weight2*entropy2)
        return entropy_list


    def _gini_split(self,featIdx,y,X_column, split_thresh):
        # generate splits
        left_idxs, right_idxs = self._split(featIdx,X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            # this should be set to a very large value, so the greedy algorithm could not use this as the best gini_split
            # if this value is set to 0,then all value returned are
            return 1
        # weighted gini
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = gini(getRows(y,left_idxs)), gini(getRows(y,right_idxs))
        gini_split = (n_l/n) * g_l + (n_r/n) * g_r

        return gini_split


    def _split(self,featIdx,X_column, split_thresh):
        if featIdx in continuousFeaturesIndices:
            # return the splited values with 1 dim

            #left_idxs = [i for i in X_column if i <=split_thresh]
            left_idxs = [idx for idx,val in enumerate(X_column) if float(val) <= float(split_thresh)]
            right_idx = [idx for idx,val in enumerate(X_column) if float(val) > float(split_thresh)]
            # print("X_column")
            # print(X_column)
            # print("split_thresh")
            # print(split_thresh)
            # print("featIdx")
            # print(featIdx)
            # print("left_idx:")
            # print(left_idxs)
            # print("right_idx:")
            # print(right_idx)

        else:
            left_idxs = [idx for idx,val in enumerate(X_column) if val == split_thresh]
            right_idx = [idx for idx,val in enumerate(X_column) if val != split_thresh]
        return left_idxs, right_idx



    def _most_common_label(self,y):
        #print(y)
        counter = Counter(y)
        #print(y)
        # return two tuples, want to have the first element of the list. to get the most common value
        most_commoon = counter.most_common(1)[0][0]
        return most_commoon

    def predict(self,X):
        # traverse the tree and get the result
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self,x,node):
        # check the stopping criterion
        if node.is_leaf():
            return node.value

        if node.feature in continuousFeaturesIndices:
            # return the splited values with 1 dim
            if float(x[node.feature]) <= float(node.threshold):
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] == str(node.threshold):
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
