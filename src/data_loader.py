import pandas as pd


class DataLoader:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
        df_train = pd.read_csv(self.train_file, index_col=0)
        df_test = pd.read_csv(self.test_file, index_col=0)
        return df_train, df_test
