import numpy as np


class Datasets(object):
    def __init__(self, df):
        """

        :param df:
        """
        self.df = df

    @property
    def review_matrix(self):
        return self.df.to_numpy()

    @property
    def users(self):
        return self.df.columns.values

    @property
    def posts(self):
        return self.df.index.values

    def train_test_split(self, test_size):
        """

        :param test_size:
        :return:
        """
        train_matrix = self.review_matrix.copy()
        test_matrix = self.review_matrix.copy()

        mask = ~np.isnan(self.review_matrix)

        n_posts, n_users = mask.shape
        for post_idx in range(n_posts):
            for user_idx in range(n_users):
                if mask[post_idx, user_idx]:
                    take_it_to_test = np.random.choice(a=[False, True], p=[1 - test_size, test_size])
                    if take_it_to_test:
                        train_matrix[post_idx, user_idx] = np.nan
                    else:
                        test_matrix[post_idx, user_idx] = np.nan
        return train_matrix, test_matrix
