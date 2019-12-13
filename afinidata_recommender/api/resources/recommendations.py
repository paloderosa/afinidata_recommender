
from sqlalchemy import create_engine

from recommender.models import CollaborativeFiltering
from recommender.read_db import ReadDatabase

# set up environment variables
load_dotenv('.env')

# set up database reader
DB_URI = os.environ.get("DB_URI")
engine = create_engine(DB_URI)

reader = ReadDatabase(engine, 'CM_BD')

question_df = reader.get_data('id, post_id', 'posts_question', None).set_index('id')
taxonomy_df = reader.get_data('post_id, area_id', 'posts_taxonomy', None)
taxonomy_areas = taxonomy_df.groupby('area_id')


# model initialization and load
model = CollaborativeFiltering()

model.load_model('afinidata_recommender_model_specs')

ratings = model.afinidata_recommend(user_id=5, question_d