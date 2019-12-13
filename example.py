import pandas as pd

from tasks import recommend, train, refresh_data


fresh_data = refresh_data.delay()

if fresh_data.ready():
    recommendations = recommend.delay(5, 10, fresh_data.get(timeout=1))
    recommendations_df = pd.read_json(recommendations.get(timeout=1))