from src.algorithm_evaluator import AlgorithmEvaluator
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.knn import KNN
from src.user_input_predictor import UserInputPredictor

k = 3

train_file = '../data/iristrain.csv'
test_file = '../data/iristest.csv'


def main():
    data_loader = DataLoader(train_file, test_file)
    df_train, df_test = data_loader.load_data()
    data_cleaner = DataCleaner(df_train, df_test)
    train, test = data_cleaner.to_numpy()
    knn = KNN(train, k)
    algorithm_evaluator = AlgorithmEvaluator(knn, test)
    algorithm_evaluator.evaluate_model()
    user_input_predictor = UserInputPredictor(knn, test)
    user_input_predictor.predict_user_input()


if __name__ == '__main__':
    main()
