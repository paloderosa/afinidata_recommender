import numpy as np
import pandas as pd
import pickle


class CollaborativeFiltering(object):
    """
    Collaborative filtering recommender system.
    """

    parameters = {}
    losses = {}
    hyperparameters = {}
    has_been_trained = False
    actors = {}
    n_users = 0
    n_items = 0

    def __init__(self):
        self.name = 'Collaborative Filtering'

    def _initialize_model(self, n_features, alpha):
        self.parameters = {
            'mean_rating': 0.001 * np.random.rand(1, 1),
            'bias_user': 0.001 * np.random.rand(1, self.n_users),
            'bias_item': 0.001 * np.random.rand(self.n_items, 1),
            'feat_vec': 0.001 * np.random.rand(n_features, self.n_items),
            'user_vec': 0.001 * np.random.rand(n_features, self.n_users)
        }
        self.losses = {
            'train': np.array([]),
            'test': np.array([])
        }
        self.hyperparameters = {
            'regularization': alpha,
            'latent features': n_features
        }

    def _loss_root(self, review_matrix, mu, b_user, b_item, x, theta):
        return mu + b_user + b_item + np.dot(x.T, theta) - review_matrix

    def loss(self, review_matrix, mu, b_user, b_item, x, theta, alpha):
        return (1 / 2.) * np.nansum(np.square(mu + b_user + b_item + np.dot(x.T, theta) - review_matrix)) \
               + (alpha / 2.) * np.sum(np.square(x)) \
               + (alpha / 2.) * np.sum(np.square(theta)) \
               + (alpha / 2.) * np.sum(np.square(b_item)) \
               + (alpha / 2.) * np.sum(np.square(b_user))

    def predict(self, mu, b_user, b_item, x, theta):
        return mu + b_user + b_item + np.dot(x.T, theta)

    def predict_default(self, mu, b_item):
        return mu + b_item

    def verify_gradients(self, review_matrix, alpha, n_features, lr):
        mu = 0.001 * np.random.rand(1, 1)
        b_user = 0.001 * np.random.rand(1, self.n_users)
        b_item = 0.001 * np.random.rand(self.n_items, 1)
        x = 0.001 * np.random.rand(n_features, self.n_items)
        theta = 0.001 * np.random.rand(n_features, self.n_users)

        mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad = self.parameter_gradients(
            review_matrix, mu, b_user, b_item, x, theta, alpha, n_features
        )

        mu_adv = mu + lr

        b_user_adv = b_user.copy()
        b_user_adv[0, 5] += lr

        b_item_adv = b_item.copy()
        b_item_adv[5, 0] += lr

        mu_grad_def = (self.loss(review_matrix, mu_adv, b_user, b_item, x, theta, alpha) -
                       self.loss(review_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        b_user_grad_def = (self.loss(review_matrix, mu, b_user_adv, b_item, x, theta, alpha) -
                           self.loss(review_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        b_item_grad_def = (self.loss(review_matrix, mu, b_user, b_item_adv, x, theta, alpha) -
                           self.loss(review_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        print(f'Gradient method: mu     {mu_grad:.5f} / gradient definition {mu_grad_def:.5f} / difference {mu_grad - mu_grad_def}')
        print(f'Gradient method: b_user {b_user_grad[0,5]:.5f} / gradient definition {b_user_grad_def:.5f} / difference {b_user_grad[0,5] - b_user_grad_def}')
        print(f'Gradient method: b_item {b_item_grad[5,0]:.5f} / gradient definition {b_item_grad_def:.5f} / difference {b_item_grad[5,0] - b_item_grad_def}')

    def parameter_gradients(self, review_matrix, mu, b_user, b_item, x, theta, alpha, n_features):
        mask = ~np.isnan(review_matrix)
        n_posts, n_users = review_matrix.shape

        x_grad = np.zeros(x.shape)
        theta_grad = np.zeros(theta.shape)

        loss_root = mu + b_item + b_user + np.dot(x.T, theta) - review_matrix

        mu_grad = np.nansum(loss_root)

        b_user_grad = np.nansum(loss_root, axis=0).reshape(1, -1) + alpha * b_user

        b_item_grad = np.nansum(loss_root, axis=1).reshape(-1, 1) + alpha * b_item

        # run over posts
        for post in range(n_posts):
            # select users that reviewed post
            idx = mask[post, :]
            # restrict Y and Theta to users who reviewed post
            review_matrix_temp = review_matrix[post, idx].reshape(1, -1)
            theta_temp = theta[:, idx].reshape(n_features, -1)
            x_temp = x[:, post].reshape(n_features, -1)
            x_grad[:, post] = np.dot(theta_temp, np.dot(theta_temp.T, x_temp) - review_matrix_temp.T)\
                .reshape(n_features)
        x_grad = x_grad + alpha * x

        # run over users
        for user in range(n_users):
            idx = mask[:, user]
            # restrict Y, X  and Theta to posts reviewed by user
            review_matrix_temp = review_matrix[idx, user].reshape(-1, 1)
            theta_temp = theta[:, user].reshape(n_features, -1)
            x_temp = x[:, idx].reshape(n_features, -1)
            theta_grad[:, user] = np.dot(x_temp, np.dot(x_temp.T, theta_temp) - review_matrix_temp)\
                .reshape(n_features)
        theta_grad = theta_grad + alpha * theta

        return mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad

    def train(self, resume, train_matrix, test_matrix, epochs, alpha, n_features, lr):
        if not resume:
            self._initialize_model(n_features, alpha)
            self.has_been_trained = True
        else:
            assert self.has_been_trained, 'The model has not been trained or loaded'
            alpha = self.hyperparameters['regularization']
            n_features = self.hyperparameters['latent features']

        n_train = np.count_nonzero(~np.isnan(train_matrix))
        n_test = np.count_nonzero(~np.isnan(test_matrix))

        print('*' * 80)
        print(f'training recommendation model for {epochs} epochs with learning rate {lr} and \n' +
              f'hyperparameters regularization: {alpha} / latent features: {n_features}')
        print('*' * 80)
        for epoch in range(epochs):
            mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad = self.parameter_gradients(
                review_matrix=train_matrix,
                mu=self.parameters['mean_rating'],
                b_user=self.parameters['bias_user'],
                b_item=self.parameters['bias_item'],
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=alpha,
                n_features=n_features
            )

            self.parameters['mean_rating'] -= lr * mu_grad
            self.parameters['bias_user'] -= lr * b_user_grad
            self.parameters['bias_item'] -= lr * b_item_grad
            self.parameters['feat_vec'] -= lr * x_grad
            self.parameters['user_vec'] -= lr * theta_grad

            train_loss = self.loss(
                review_matrix=train_matrix,
                mu=self.parameters['mean_rating'],
                b_user=self.parameters['bias_user'],
                b_item=self.parameters['bias_item'],
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=alpha) / n_train

            self.losses['train'] = np.append(self.losses['train'], train_loss)

            test_loss = self.loss(
                review_matrix=test_matrix,
                mu=self.parameters['mean_rating'],
                b_user=self.parameters['bias_user'],
                b_item=self.parameters['bias_item'],
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=0) / n_test
            self.losses['test'] = np.append(self.losses['test'], test_loss)

            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1:05d} / train loss {train_loss:.6f} / test loss {test_loss:.6f}')
        print('*' * 80)
        print('training finished. final losses are')
        print(f'Epoch {epochs:05d} / train loss {train_loss:.6f} / test loss {test_loss:.6f}')

    def predict_rating(self, idx):
        try:
            predictions = self.predict(
                self.parameters['mean_rating'],
                self.parameters['bias_user'],
                self.parameters['bias_item'],
                self.parameters['feat_vec'],
                self.parameters['user_vec']
                )[:, idx]

        except IndexError:
            predictions = self.predict_default(
                self.parameters['mean_rating'],
                self.parameters['bias_item']
            )[:, 0]

        n_items = predictions.shape[0]

        predictions_with_id = np.zeros((n_items, 2))
        predictions_with_id[:, 1] = predictions

        return pd.DataFrame({
            'predictions': predictions_with_id[:, 1]
        })

    def save_model(self, filename):
        model_specs = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'hyperparameters': self.hyperparameters,
            'parameters': self.parameters,
            'losses': self.losses,
            'actors': self.actors
        }
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(model_specs, f)

    def load_model(self, filename):
        with open(f'{filename}.pkl', 'rb') as f:
            model_specs = pickle.load(f)
            self.n_users = model_specs['n_users']
            self.n_items = model_specs['n_items']
            self.hyperparameters = model_specs['hyperparameters']
            self.parameters = model_specs['parameters']
            self.losses = model_specs['losses']
            self.actors = model_specs['actors']
            self.has_been_trained = True

    def afinidata_recommend(self, user_id, months, data_required):
        # data is sequentially ordered and the relation between the indices and the actual
        # user_id is stored in self.actors['users']. if the user is in this list, which means
        # that this user has given at least one rating, then find it, else go to the
        # exceptional case handled by self.predict_rating.
        if user_id in self.actors['users']:
            idx, = np.where(self.actors['users'] == user_id)
            predictions = self.predict_rating(idx[0])
        else:
            predictions = self.predict_rating(-1)
        predictions['question_id'] = self.actors['items']

        question_df = data_required['question_df']
        taxonomy_df = data_required['taxonomy_df']
        content_df = data_required['content_df']
        interaction_df = data_required['interaction_df']

        # we add the columns corresponding to question_id and area_id
        predictions = pd.merge(predictions, question_df, 'inner', left_on='question_id', right_on='id')
        predictions = pd.merge(predictions, taxonomy_df, 'inner', 'post_id')

        content_for_age = content_df[(content_df['min_range'] <= months) & (content_df['max_range'] >= months)][
            'id'].values.tolist()
        sent_activities = interaction_df[interaction_df['user_id'] == user_id]['post_id'].values.tolist()

        relevant_predictions = predictions[predictions['post_id'].isin(content_for_age)]
        relevant_unseen_predictions = relevant_predictions[~relevant_predictions['post_id'].isin(sent_activities)]

        # we group the predictions by area and take the mean
        area_performance = relevant_predictions[['predictions', 'area_id']].groupby('area_id').mean()
        # we normalize the mean predictions by area
        area_performance['normalized'] = area_performance['predictions'].apply(
            lambda x: (x - area_performance['predictions'].mean()) / area_performance['predictions'].std())
        # we compute probabilities from the normalized means such that lower means correspond
        # to higher probabilities
        area_performance['probabilities'] = area_performance['normalized'].apply(lambda x: np.exp(-x))

        if len(relevant_unseen_predictions.index) == 0:
            # we randomly select an area according to the assigned probabilities
            area_performance['probabilities'] = area_performance['probabilities'] / area_performance[
                'probabilities'].sum()
            selected_area = np.random.choice(area_performance.index.values, p=area_performance['probabilities'].values)
            return relevant_predictions[
                relevant_predictions['area_id'] == selected_area
                ].sort_values('predictions', ascending=False).to_json()
        else:
            available_areas = relevant_unseen_predictions['area_id'].unique()
            area_performance = area_performance[area_performance.index.isin(available_areas)]
            area_performance['probabilities'] = area_performance['probabilities'] / area_performance[
                'probabilities'].sum()
            selected_area = np.random.choice(area_performance.index.values, p=area_performance['probabilities'].values)
            return relevant_unseen_predictions[
                relevant_unseen_predictions['area_id'] == selected_area
                ].sort_values('predictions', ascending=False).to_json()
