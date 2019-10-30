import numpy as np


class CollaborativeFiltering(object):
    """
    Collaborative filtering recommender system.
    """
    def __init__(self, n_posts, n_users, n_features):
        self.name = 'Collaborative Filtering'
        self.n_features = n_features
        self.n_users = n_users
        self.n_posts = n_posts
        self.parameters = {
            'feat_vec': 0.001 * np.random.rand(n_features, n_posts),
            'user_vec': 0.001 * np.random.rand(n_features, n_users)
        }
        self.training_losses = np.array([])
        self.test_losses = np.array([])

    def loss(self, review_matrix, x, theta, alpha):
        return (1 / 2.) * np.nansum(np.square(np.dot(x.T, theta) - review_matrix)) \
               + (alpha / 2.) * np.sum(np.square(x)) \
               + (alpha / 2.) * np.sum(np.square(theta))

    def predict(self, x, theta):
        return np.dot(x.T, theta)

    def parameter_gradients(self, review_matrix, x, theta, alpha):
        mask = ~np.isnan(review_matrix)
        x_grad = np.zeros(x.shape)
        theta_grad = np.zeros(theta.shape)
        n_posts, n_users = review_matrix.shape

        # run over posts
        for post in range(n_posts):
            # select users that reviewed post
            idx = mask[post, :]
            # restrict Y and Theta to users who reviewed post
            review_matrix_temp = review_matrix[post, idx].reshape(1, -1)
            theta_temp = theta[:, idx].reshape(self.n_features, -1)
            x_temp = x[:, post].reshape(self.n_features, -1)
            x_grad[:, post] = np.dot(theta_temp, np.dot(theta_temp.T, x_temp) - review_matrix_temp.T)\
                .reshape(self.n_features)
        x_grad = x_grad + alpha * x

        # run over users
        for user in range(n_users):
            idx = mask[:, user]
            # restrict Y, X  and Theta to posts reviewed by user
            review_matrix_temp = review_matrix[idx, user].reshape(-1, 1)
            theta_temp = theta[:, user].reshape(self.n_features, -1)
            x_temp = x[:, idx].reshape(self.n_features, -1)
            theta_grad[:, user] = np.dot(x_temp, np.dot(x_temp.T, theta_temp) - review_matrix_temp)\
                .reshape(self.n_features)
        theta_grad = theta_grad + alpha * theta

        return x_grad, theta_grad

    def train(self, train_matrix, test_matrix, epochs, alpha, lr):

        self.parameters = {
            'feat_vec': 0.001 * np.random.rand(self.n_features, self.n_posts),
            'user_vec': 0.001 * np.random.rand(self.n_features, self.n_users)
        }

        self.training_losses = np.zeros(epochs)
        self.test_losses = np.zeros(epochs)

        for epoch in range(epochs):
            x_grad, theta_grad = self.parameter_gradients(
                review_matrix=train_matrix,
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=alpha)

            self.parameters['feat_vec'] = self.parameters['feat_vec'] - lr * x_grad
            self.parameters['user_vec'] = self.parameters['user_vec'] - lr * theta_grad

            train_loss = self.loss(train_matrix, self.parameters['feat_vec'], self.parameters['user_vec'], alpha)
            self.training_losses[epoch] = train_loss
            test_loss = self.loss(test_matrix, self.parameters['feat_vec'], self.parameters['user_vec'], 0)
            self.test_losses[epoch] = test_loss

            print('Epoch {} / train loss {:.3f} / test loss {:.3f}'.format(epoch, train_loss, test_loss))

    def recommend(self, user_id):
        return -np.sort(
            -self.predict(self.parameters['feat_vec'], self.parameters['user_vec'][:, [user_id]]),
            axis=0
        )
