from numpy.random import default_rng
from metric import MyMetric
import numpy as np

def cal_entropy(labels):
    unique, counts = np.unique(labels, return_counts=True)
    pk = counts/np.sum(counts)
    return -np.sum(pk*np.log2(pk))

def cal_info_gain(labels_all, labels_split):
    n_labels_all = len(labels_all)
    n_labels_split = [len(split) / n_labels_all for split in labels_split]
    H_all = cal_entropy(labels_all)

    H_split = [n_labels_split[i]*cal_entropy(labels_split[i]) for i in range(len(labels_split))]

    remainder = np.sum(H_split)
    return H_all - remainder

def n_fold_split(data, n=10, random_generator=default_rng(seed=1024)):
    """
    A function splits the dateset into n-folds, stores and returns them as a list
    """
    assert n > 1, '# of splits must > 1'
    shuffled_index = random_generator.permutation(data.shape[0])
    index_folds = np.array_split(ary=shuffled_index, indices_or_sections=n)
    return [data[fold_idx] for fold_idx in index_folds]

def cross_validation(classifier, n_folds, prune=False):
    labels = np.unique(np.array(n_folds)[:, :, -1])
    metric = MyMetric(labels) # Initiate the metric

    # prune_fold = n_folds[0]
    # n_folds = n_folds[1:]

    # iteratively update the metric with each fold as validation set
    for i, valid_fold in enumerate(n_folds):
        print('[Training] fold %d as validation' % i)
        X_valid = valid_fold[:, :-1]
        y_valid = valid_fold[:, -1]
        # Concatenate the training set
        train_fold = n_folds[:i] + n_folds[i+1:]
        train_fold = np.concatenate(train_fold)
        X_train = train_fold[:, :-1]
        y_train = train_fold[:, -1]
        # Train Decision Tree
        tr = classifier(train_fold, 0)
        if prune:
            print('Pruning the tree')
            tr.iterative_prune(tr.root, train_fold, valid_fold)
        # Predict for the current fold
        y_pred = tr.predict_all(X_valid, tr.root)
        y_true = y_valid
        # Update the metric
        print('[Training] finished one fold, updating metric')
        metric.update(y_true,y_pred)

    return tr, metric

if __name__ == '__main__':
    data = np.loadtxt('wifi_db/noisy_dataset.txt')
    from Decision_tree import DecisionTree
    n_folds = n_fold_split(data,2,default_rng(1024))
    tr, metric = cross_validation(DecisionTree,n_folds,True)
    print(metric.get_metric())
    # Test entropy (from lecture example)
    sudo_labels_all = np.array([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2])
    sudo_labels_split = [np.array([1,1,1,1]), np.array([1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2])]
    eps = 0.0001
    assert cal_entropy(sudo_labels_all) - 0.9928 <= eps
    assert cal_info_gain(sudo_labels_all,sudo_labels_split) - 0.2760 <= eps




