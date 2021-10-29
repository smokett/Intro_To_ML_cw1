import numpy as np

class MyMetric:
    def __init__(self, labels):
        self.metric = {}
        self.counter = 0
        self.labels = labels
        self.n_labels = len(self.labels)

    def confusion_matrix(self, y_true, y_pred):
        """
        A function to compute confusion matrix
        """
        cm = np.zeros((self.n_labels, self.n_labels))
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                cm[i][j] = np.sum(np.logical_and(y_true == self.labels[i], y_pred == self.labels[j]))
        return cm 
    
    def accuracy(self, y_true, y_pred):
        ac = np.zeros(self.n_labels)
        for i, label in enumerate(self.labels):
            TP = np.sum(np.logical_and(y_pred == label, y_true == label))
            FP = np.sum(np.logical_and(y_pred == label, y_true != label))
            FN = np.sum(np.logical_and(y_pred != label, y_true == label))
            TN = np.sum(np.logical_and(y_pred != label, y_true != label))
            a = (TP+TN)/(TP+FP+FN+TN) if (TP+FP+FN+TN) != 0 else 0
            ac[i] = a
        return ac



    def precision(self, y_true, y_pred):
        """
        A function to compute precision score
        """
        precision = np.zeros(self.n_labels)
        for i, label in enumerate(self.labels):
            TP = np.sum(np.logical_and(y_pred == label, y_true == label))
            FP = np.sum(np.logical_and(y_pred == label, y_true != label))
            p = TP/(TP+FP) if (TP+FP)!=0 else 0
            precision[i] = p
        return precision

    def recall(self, y_true, y_pred):
        """
        A function to compute recall score
        """
        recall = np.zeros(self.n_labels)
        for i, label in enumerate(self.labels):
            TP = np.sum(np.logical_and(y_pred == label, y_true == label))
            FN = np.sum(np.logical_and(y_pred != label, y_true == label))
            r = (TP)/(TP+FN) if (TP+FN)!=0 else 0 
            recall[i] = r
        return recall

    def f1(self, precision, recall):
        """
        A function to compute F1 score
        """
        # Avoid broadcast here to deal with 0 division
        result = np.zeros(len(precision))
        for i in range(len(precision)):
            if precision[i] + recall[i] == 0:
                result[i] = 0
            else:
                result[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])
        return result
    
    

    def running_mean(self, k, prev, now):
        return prev + 1/k * (now-prev)

    def update(self, y_true, y_pred):
        """
        A function update the cumulative metric for the given fold
        """
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1 = self.f1(precision, recall)
        cm = self.confusion_matrix(y_true, y_pred)
        accuracy = self.accuracy(y_true, y_pred)
        # Now we need to store each result for comparison in inner cv
        if self.counter == 0:
            self.counter += 1
            self.metric['accuracy'] = [accuracy]
            self.metric['precision'] = [precision]
            self.metric['recall'] = [recall]
            self.metric['f1'] = [f1]
            self.metric['cm'] = [cm]
        else:
            self.counter += 1
            self.metric['accuracy'].append(accuracy)
            self.metric['precision'].append(precision)
            self.metric['recall'].append(recall)
            self.metric['f1'].append(f1)
            self.metric['cm'].append(cm)

    def get_raw_metric(self):
        return self.metric

    def get_metric(self):
        """
        A function to return the average metrics
        """
        for k, v in self.metric.items():
            # Mean over folds
            if k != 'cm':
                self.metric[k] = np.mean(v, axis=0)
            else:
                self.metric[k] = np.sum(v, axis=0)
        return self.metric, self.labels



if __name__ == '__main__':
    # Test precision/recall/confusion_matrix
    y_pred = np.array([7, 2, 7, 3, 7])
    y_true = np.array([7, 2, 7, 2, 3])
    labels = np.array([2,3,7])
    metric = MyMetric(labels)
    metric.update(y_true, y_pred)
    metric.update(y_true, y_pred)
    result, labels = metric.get_metric()
    assert result['precision'][0] == 1
    assert result['precision'][1] == 0
    assert result['precision'][2] == 2/3
    assert result['recall'][0] == 1/2
    assert result['recall'][1] == 0
    assert result['recall'][2] == 1
    
    
    assert result['cm'][0][0] / np.sum(result['cm'][0]) == result['recall'][0]
    assert result['cm'][2][2] / np.sum(result['cm'][:,2]) == result['precision'][2]

