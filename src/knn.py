from math import sqrt
import numpy as np


def calculate_euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1) - 1):
        distance += (x1[i] - x2[i]) ** 2
    return sqrt(distance)


class KNN:
    def __init__(self, train, k):
        self.train = train
        self.k = k

    def get_nearest_neighbors(self, test_row):
        distances = []
        data = []
        for train_row in self.train:
            distance = calculate_euclidean_distance(test_row, train_row)
            distances.append(distance)
            data.append(train_row)

        distances = np.array(distances)
        data = np.array(data)
        distances_indices_sorted = distances.argsort()
        data = data[distances_indices_sorted]
        neighbors = data[:self.k]
        return neighbors

    def predict_classification(self, test_row):
        neighbors = self.get_nearest_neighbors(test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(output_values, key=output_values.count)
        return prediction
