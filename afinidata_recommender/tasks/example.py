from afinidata_recommender.extensions import celery


@celery.task
def dummy_task():
    return "OK"
