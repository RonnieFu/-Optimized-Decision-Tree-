import numpy as np
from collections import Counter

def entropy(y):
    counter = Counter(y)
    #hist = np.bincount(y)
    ps = [counter.most_common()[i][1]/len(y) for i in range(len(counter.most_common()))]
    return -np.sum([p * np.log2(p) for p in ps])

def gini(y):
    counter = Counter(y)
    ps = [counter.most_common()[i][1]/len(y) for i in range(len(counter.most_common()))]
    return 1-np.sum([p**2 for p in ps])

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
    def __init__(self,min_samples_split=2,max_depth=10,n_feats=None,criterion="entropy"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_feats = n_feats
        # later need to know the root
        self.root = None
    def fit(self,X,y):
        # training a DT
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # select random features, the returned value should be a list
        feat_idxs = np.random.choice(n_features,self.n_feats,replace=False)

        # greedy search to find the best split
        best_feat, best_thresh = self._best_criteria(X,y,feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        # TODO check the splits
        #print(f"best_feat: {best_feat}, best_thresh: {best_thresh}")
        #print(f"left_idxs:: {left_idxs}, right_idxs: {right_idxs}")
        # put only the rows of left_idxs and all features, for y only left idxs
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self,X,y,feat_idxs):
        split_idx, split_thresh = None, None
        if self.criterion == "entropy":
            best_gain = -1
            for feat_idx in feat_idxs:
            # every time we try one feature
                # get the selected column as the split
                X_column = X[:,feat_idx]
                thresholds = np.unique(X_column) # get all unique values
                for threshold in thresholds:
                    gain = self._information_gain(y,X_column, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = threshold

        elif self.criterion == "gini":
            smallest_gini_split = 1
            # two for loops to traverse all features and all values
            for feat_idx in feat_idxs:
                X_column = X[:,feat_idx]
                thresholds = np.unique(X_column) # get all unique values
                for threshold in thresholds:
                    # in gini index the best gain is the smallest gini
                    gini_split = self._gini_split(y,X_column, threshold)
                    if gini_split < smallest_gini_split:
                        smallest_gini_split = gini_split
                        split_idx = feat_idx
                        split_thresh = threshold
            #print("smallest_gini_split:",smallest_gini_split,split_idx,split_thresh)
        return split_idx, split_thresh

    def _information_gain(self,y,X_column, split_thresh):
        # parent Entropy
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)


        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted avg child Entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # return ig
        ig = parent_entropy - child_entropy
        return ig

    def _gini_split(self,y,X_column, split_thresh):
        # generate splits
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            # this should be set to a very large value, so the greedy algorithm could not use this as the best gini_split
            # if this value is set to 0,then all value returned are
            return 1
        # weighted gini
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = gini(y[left_idxs]), gini(y[right_idxs])
        gini_split = (n_l/n) * g_l + (n_r/n) * g_r

        # return gini_split

        return gini_split


    def _split(self,X_column, split_thresh):
        if type(X_column[0]) != np.str:
            # return the splited values with 1 dim
            left_idxs = np.argwhere(X_column <= split_thresh).flatten()
            right_idx = np.argwhere(X_column > split_thresh).flatten()

        else:
            left_idxs = np.argwhere(X_column == split_thresh).flatten()
            right_idx = np.argwhere(X_column != split_thresh).flatten()

        return left_idxs, right_idx
    def _most_common_label(self,y):
        #print(y)
        counter = Counter(y)
        # return two tuples, want to have the first element of the list. to get the most common value
        most_commoon = counter.most_common(1)[0][0]
        return most_commoon

    def predict(self,X):
        # traverse the tree and get the result
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self,x,node):
        # check the stopping criterion
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


