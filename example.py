import pandas as pd
import pickle

from tasks import recommend, train, refresh_data

fresh_data = None
refresh_data.delay('fresh_data')
with open(f'fresh_data.pkl', 'rb') as f:
    fresh_data = pickle.load(f)

# model initialization and load

recommendations = recommend.delay(5, 10, fresh_data)
recommendations_df = pd.read_json(recommendations.get(timeout=1))