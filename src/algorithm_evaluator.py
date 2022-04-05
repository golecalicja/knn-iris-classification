import numpy as np


def get_correct_predictions(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct


class AlgorithmEvaluator:
    def __init__(self, knn, test):
        self.knn = knn
        self.test = test

    def evaluate_model(self):
        correct, accuracy = self.get_test_set_predictions()
        print('Correct predictions: %d' % correct)
        print('Accuracy: {:.1%}'.format(accuracy))

    def get_test_set_predictions(self):
        actual = self.test[:, -1]
        predicted = []

        for test_row in self.test:
            prediction = self.knn.predict_classification(test_row)
            predicted = np.append(predicted, prediction)

        correct = get_correct_predictions(actual, predicted)
        accuracy = correct / len(actual)

        return correct, accuracy
