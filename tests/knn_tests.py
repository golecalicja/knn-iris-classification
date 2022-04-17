import numpy as np

from src.knn import calculate_euclidean_distance, KNN

train = [[1, 1, 1, 1, 'setosa'], [2, 2, 2, 2, 'setosa'], [3, 3, 3, 3, 'setosa']]
k = 1
knn = KNN(train, k)


def test_calculate_euclidean_distance():
    # given
    x1 = [1, 1, 1, 'setosa']
    x2 = [2, 1, 1, 'virginica']
    # when
    result = calculate_euclidean_distance(x1, x2)
    # then
    assert result == 1


def test_get_nearest_neighbors():
    # given
    test_row = [1, 1, 1, 1, 'setosa']
    # when
    result = knn.get_nearest_neighbors(test_row)
    # then
    assert np.array_equal(result, [[1, 1, 1, 1, 'setosa']])


def test_predict_classification():
    # given
    test_row = [1, 1, 1, 1, 'setosa']
    # when
    result = knn.predict_classification(test_row)
    # then
    assert result == 'setosa'
