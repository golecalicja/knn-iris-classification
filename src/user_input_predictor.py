import numpy as np


class UserInputPredictor:
    def __init__(self, knn, test):
        self.knn = knn
        self.test = test

    def predict_user_input(self):
        prediction = self.knn.predict_classification(self.get_user_input())
        print('Predicted iris: ' + prediction)

    def get_user_input(self):
        print('Enter vector values: ')
        vector = []
        size = self.test[0].size - 1
        for i in range(size):
            vector.append(float(input('Value: ')))
        vector = np.array(vector)
        return vector
