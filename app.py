import os

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

from recommender.read_db import ReadDatabase
from recommender.preprocess import SetUpDataframes


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

DB_URI = os.environ.get("DB_URI")
engine = create_engine(DB_URI)

reader = ReadDatabase(engine, 'CM_DB')

feedback_df = reader.get_data('user_id, post_id, value', 'posts_feedback', None)
feedback_matrix = SetUpDataframes.feedback_matrix(feedback_df)









