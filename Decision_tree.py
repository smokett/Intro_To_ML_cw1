import numpy as np
from utils import cal_entropy, cal_info_gain, cross_validation, n_fold_split

class Node:
    def __init__(self, attribute, value, left, right, is_leaf):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf



class DecisionTree:
    def __init__(self, training_dataset, depth=0):
        self.root, self.depth = self.decision_tree_learning(training_dataset,depth)

    def decision_tree_learning(self, training_dataset, depth):
        """
        A recursive function to build decision tree
        """
        # Base case, when encounters pure dataset
        if len(np.unique(training_dataset[:,-1]))==1:
            return (Node(self.majority_vote(training_dataset),None, None, None, True), depth)
        else:
            feature_id, value, d_left, d_right = self.find_split(training_dataset)
            # Stop if all samples are on the same side
            if len(d_left) == 0 or len(d_right) == 0:
                return (Node(self.majority_vote(training_dataset),None,None,None, True), depth)
            l_branch,l_depth = self.decision_tree_learning(d_left, depth+1)
            r_branch,r_depth = self.decision_tree_learning(d_right, depth+1)
            return (Node(feature_id, value, l_branch, r_branch, False), max(l_depth,r_depth))

    def find_split(self, training_dataset):
        """
        A function to split the node
        """
        best_ig = -1
        for i in range(training_dataset.shape[1]-1):
            feature = training_dataset[:, i]
            labels = training_dataset[:, -1]

            # Sort feature and labels by feature value from low to high
            feature_sort_idx = np.argsort(feature)
            feature_sorted = feature[feature_sort_idx]
            labels_sorted = labels[feature_sort_idx]

            # List to store all checked points, we DONT have to check same split point twice!
            checked_split_points = []
            for j in range(len(labels_sorted)-1):
                # When there is a change of label, we try to split
                if labels_sorted[j+1] != labels_sorted[j]:
                    mid_point = (feature_sorted[j] + feature_sorted[j+1])/2
                    if mid_point not in checked_split_points:
                        # We have checked this split point!
                        checked_split_points.append(mid_point)
                        # We need to be explict in case when feature value ties at split point
                        l_labels = labels_sorted[feature_sorted<mid_point]
                        r_labels = labels_sorted[feature_sorted>=mid_point]
                        s_labels = [l_labels, r_labels]

                        # Calculate information gain
                        ig = cal_info_gain(labels_sorted, s_labels)

                        # Find best ig and record the relavant information
                        if ig > best_ig:
                            best_ig = ig
                            best_feature_id = i
                            best_value = mid_point
                            d_left = training_dataset[feature_sort_idx[feature_sorted<best_value]]
                            d_right = training_dataset[feature_sort_idx[feature_sorted>=best_value]]
        return best_feature_id, best_value, d_left, d_right

    def majority_vote(self, training_dataset):
        """
        A function to handle leaf node
        """
        unique, counts = np.unique(training_dataset[:, -1], return_counts=True)
        return unique[np.argmax(counts)]

    def evaluate_error(self, valid_set, node):
        X = valid_set[:, :-1]
        y_true = valid_set[:, -1]
        y_pred = self.predict_all(X, node)
        return np.sum(y_true!=y_pred)

    def prune(self, root, train_set, valid_set):
        if root.left.is_leaf and root.right.is_leaf:
            if len(valid_set) >0:
                valid_error = self.evaluate_error(valid_set, root)
                voting_label = self.majority_vote(train_set)
                voting_valid_error = np.sum(valid_set[:,-1] != voting_label)

                if valid_error > voting_valid_error:
                    root.left = None
                    root.right = None
                    root.attribute = voting_label
                    root.value = None
                    root.is_leaf = True
                    
        else:
            # Divide data into two branches
            l_valid_data = valid_set[valid_set[:, root.attribute] < root.value]
            r_valid_data = valid_set[valid_set[:, root.attribute] >= root.value]
            l_train_data = train_set[train_set[:, root.attribute] < root.value]
            r_train_data = train_set[train_set[:, root.attribute] >= root.value]
            
            # Left node is leaf, try to prune right node
            if root.left.is_leaf and not root.right.is_leaf:
                self.prune(root.right, r_train_data, r_valid_data)
            # Right node is leaf, try to prune left node
            elif not root.left.is_leaf and root.right.is_leaf:
                self.prune(root.left, l_train_data, l_valid_data)
            # Both nodes are not leaves, try to prune both nodes
            else:
                self.prune(root.left, l_train_data, l_valid_data)
                self.prune(root.right, r_train_data, r_valid_data)
                


    def iterative_prune(self, root, train_set, valid_set):

        # Each time, we prune, it is a guarenteed improvement
        while True:
            self.prune(root, train_set, valid_set)
            prev_root = root
            prev_error = self.evaluate_error(valid_set, prev_root)

            self.prune(prev_root, train_set, valid_set)
            now_error = self.evaluate_error(valid_set, prev_root)
            #print('  now_error:{}, prev_error:{}'.format(now_error, prev_error))
            if now_error == prev_error:
                break

    def predict_all(self, X, node):
        """
        A function to predict batch inputs
        """
        return np.array([self.predict(x, node) for x in X])

    def predict(self, x, node):
        """
        A function to predict one input
        """

        if node.is_leaf:
            return int(node.attribute)
        if x[node.attribute] < node.value:
            return self.predict(x, node.left)
        else:
            return self.predict(x, node.right)




if __name__ == '__main__':

    # Test at least we are overfitting
    data = np.loadtxt('wifi_db/clean_dataset.txt')
    t = DecisionTree(data,0)
    tree = t.root
    dummy_test_input = np.array([-64, -56, -61, -66, -71, -82, -81])
    assert t.predict(dummy_test_input, t.root) == 1
    dummy_test_input = np.array([-54, -59, -52, -63, -62, -76, -81])
    assert t.predict(dummy_test_input, t.root) == 3
    t.prune(tree, data)



