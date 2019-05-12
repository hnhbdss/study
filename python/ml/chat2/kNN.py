from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):

    data_set_size = dataSet.shape[0]
    diff_mat= tile(inX, (data_set_size, 1)) - dataSet
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis = 1)
    distances = sq_distances ** 0.5
    sorted_dist_indices = distances.argsort()
    class_counts = {}
    for i in range(k) :
        category = labels[sorted_dist_indices[i]]
        class_counts[category] = class_counts.get(category, 0) + 1
        sorted_class_count = sorted(class_counts.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

if __name__ == "__main__":
    dataset = createDataSet()
    print(classify0([0,0], dataset[0], dataset[1], 3))

