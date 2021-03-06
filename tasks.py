import logging
import os
import pickle

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import create_engine

from recommender.read_db import ReadDatabase
from recommender.models import CollaborativeFiltering
from recommender.preprocess import SetUpDataframes
from recommender.datasets import Datasets


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

load_dotenv('.env')

# environment variables
DB_URI = os.environ.get("DB_URI")
CELERY_BROKER = os.environ.get('CELERY_BROKER')
CELERY_BACKEND = os.environ.get('CELERY_BACKEND')

# set up database reader
engine = create_engine(DB_URI)
reader_cm = ReadDatabase(engine, 'CM_BD')

app = Celery('tasks', backend=CELERY_BACKEND, broker=CELERY_BROKER)


@app.task
def refresh_data(filename):
    question_df = reader_cm.get_data(
        'id, post_id',
        'posts_question',
        None)

    taxonomy_df = reader_cm.get_data(
        'post_id, area_id',
        'posts_taxonomy',
        None)

    content_df = reader_cm.get_data(
        'id, min_range, max_range',
        'posts_post',
        "status IN ('published')")

    interaction_df = reader_cm.get_data(
        'user_id, post_id',
        'posts_interaction',
        "type IN ('sended', 'sent', 'dispatched')")
    interaction_df = interaction_df[~interaction_df['post_id'].isna()]
    interaction_df['post_id'] = interaction_df['post_id'].astype('int32')

    response_df = reader_cm.get_data(
        'user_id, response, question_id',
        'posts_response',
        "created_at >= '2019-09-20'",
        None)
    response_df = response_df[
        (response_df['response'].apply(lambda x: x.isdigit())) & (response_df['response'] != '0')]
    response_df = response_df.drop_duplicates().reset_index(drop=True)

    fresh_data = {
        'question_df': question_df.to_json(),
        'taxonomy_df': taxonomy_df.to_json(),
        'content_df': content_df.to_json(),
        'interaction_df': interaction_df.to_json(),
        'response_df': response_df.to_json()
    }

    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(fresh_data, f)


@app.task
def train(epochs=10000, lr=0.00001, alpha=0., depth=1):
    # extract data from posts_response into a pandas dataframe and
    # slightly process only relevant data for training
    # in this case, so far we are only considering data for which
    # there is an alpha value in the 'response' column
    try:
        response_df = reader_cm.get_data(
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
    train_set, test_set = datasets.train_test_split(0.0)

    # model initialization
    model = CollaborativeFiltering()
    model.actors = {
        'users': response_matrix.columns.values,
        'items': response_matrix.index.values
    }
    model.n_items = len(datasets.posts)
    model.n_users = len(datasets.users)

    model.train(
        train_matrix=train_set,
        test_matrix=test_set,
        epochs=epochs,
        alpha=alpha,
        n_features=depth,
        lr=lr,
        resume=False
    )

    print('*' * 80)
    model.save_model(f'afinidata_recommender_model_specs')
    print(f'model has been saved to afinidata_recommender_model_specs.pkl in the local directory')


@app.task
def recommend(user_id, months, data_required):
    model = CollaborativeFiltering()

    model.load_model('afinidata_recommender_model_specs')

    return model.afinidata_recommend(user_id=user_id, months=months, data_required=data_required)
