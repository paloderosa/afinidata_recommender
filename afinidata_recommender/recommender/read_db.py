import pandas as pd
from sqlalchemy import text


class ReadDatabase(object):
    def __init__(self, engine, db):
        """
        Interface for reading database tables from a given database. Only simple selections
        can be made using this class. Any processing should be encapsulated elsewhere.
        TODO: select from a particular user or a particular activity should be implemented
        :param engine: sqlalchemy engine
        :param db: db name
        """
        self.engine = engine
        self.db = db

    def get_data(self, sql_query_columns, table, filter=None, index=None):
        connection = self.engine.connect()
        if filter is None:
            filter_text = ''
        else:
            filter_text = f'WHERE {filter}'
        query = text(f'SELECT {sql_query_columns} FROM {self.db}.{table} {filter_text}')
        print('-'*70 + '\n'
              + f'reading columns {sql_query_columns} from table {table} from database {self.db}')
        df = pd.read_sql(query, connection, index_col=index)
        connection.close()
        return df
