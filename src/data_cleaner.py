class DataCleaner:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def to_numpy(self):
        train = self.df_train.to_numpy()
        test = self.df_test.to_numpy()
        return train, test
