import argparse
from datetime import datetime
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

from recommender.read_db import ReadDatabase
from recommender.preprocess import SetUpDataframes
from recommender.models import CollaborativeFiltering
from recommender.datasets import Datasets


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='afinidata recommendation system training phase')
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=10000)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.00001)
parser.add_argument('--alpha', type=float, help='Regularization constant', default=0)
parser.add_argument('--depth', type=int, help='Number of latent features', default=1)
args = parser.parse_args()

if __name__ == '__main__':

    # set up environment variables
    load_dotenv('.env')

    # set up database reader
    DB_URI = os.environ.get("DB_URI")
    engine = create_engine(DB_URI)

    reader = ReadDatabase(engine, 'CM_BD')

    # extract data from posts_response into a pandas dataframe and
    # slightly process only relevant data for training
    # in this case, so far we are only considering data for which
    # there is an alpha value in the 'response' column
    try:
        response_df = reader.get_data(
            'user_id, response, question_id', 'posts_response',
            "created_at >= '2019-09-20'",
            None)
    except Exception as e:
        logger.exception("Unable to retrieve data from the database. Check your internet connection " +
                         "to the database or the parameters in the SQL query. Error: %s", e)

    response_df = response_df[
        (response_df['response'].apply(lambda x: x.isdigit())) & (response_df['response'] != '0')]
    response_df = response_df.drop_duplicates().reset_index(drop=True)
    print('*' * 80)
    print(f'total number of responses in response_df: {len(response_df)}')

    # create matrix for training with items over rows and users over columns
    # as a numpy matrix
    response_matrix = SetUpDataframes.response_matrix(response_df)

    # train test split
    datasets = Datasets(response_matrix)
    train, test = datasets.train_test_split(0.20)

    # model initialization
    model = CollaborativeFiltering()
    model.actors = {
        'users': response_matrix.columns.values,
        'items': response_matrix.index.values
    }
    model.n_items = len(datasets.posts)
    model.n_users = len(datasets.users)

    model.train(
        train_matrix=train,
        test_matrix=test,
        epochs=args.epochs,
        alpha=args.alpha,
        n_features=args.depth,
        lr=args.lr,
        resume=False
    )

    today_string = datetime.strftime(datetime.today().date(),'%Y%m%d')

    print('*' * 80)
    # model.save_model(f'afinidata_recommender_{today_string}')
    # print(f'model has been saved to afinidata_recommender_{today_string}.pkl in the local directory')
    model.save_model(f'afinidata_recommender_model_specs')
    print(f'model has been saved to afinidata_recommender_model_specs.pkl in the local directory')
