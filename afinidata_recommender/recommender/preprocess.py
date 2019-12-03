from collections import OrderedDict
from datetime import datetime
import pickle

import pandas as pd
from pandas.api.types import CategoricalDtype


class PreprocessGenericData(object):
    def __init__(self, pipeline):
        """
        Defines a list of preprocessing tasks to be performed on a Pandas dataframe with a
        temporal nature. The rows in the dataframe correspond to `events` classified according to
        different `types`.
        :param pipeline: OrderedDict of the form:
            task: argument
        where task must be a method in the class and argument must be the method's arguments.
        """
        for task in pipeline:
            assert hasattr(self, task), '{} is not a method in the class.'.format(task)
        self.pipeline = OrderedDict(pipeline)

    _basic_dtypes = ['int64', 'float64', 'datetime64', 'category']

    @classmethod
    def set_dtypes(cls, df, dtype_specs):
        """
        Transform dataframe columns into their corresponding types. This is necessary,
        for example, when a CSV file is imported and datetime columns are in a string format.
        :param df: Pandas dataframe
        :param dtype_specs: list of tuples with entries
            (column, dtype)
        where column is a string specifying the name of a dataframe column and  dtype is another
        string specifying the expected type the elements in the column should have. Supported
        dtype strings are drawn from Pandas basic dtypes:
        # https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-dtypes

        :return: transformed dataframe.
        """

        for (column, dtype) in dtype_specs:
            assert column in df.columns, '{} is not a column in the dataframe'.format(column)
            assert dtype in cls._basic_dtypes, '{} is not a (basic) supported type for conversion'.format(dtype)

            try:
                df[column] = df[column].astype(dtype, errors='ignore')
            except ValueError as ve:
                print(ve)

        return df

    @staticmethod
    def replace_in_column(df, replace_specs):
        """
        Substitute values in particular dataframe columns
        :param df: Pandas dataframe
        :param replace_specs: list of tuples with entries of the form
             (column, [(original1, substitution1), (original2, substitution2), ...])
        :return: dataframe with substitutions
        """

        for column, substitutions in replace_specs:
            assert column in df.columns, '{} is not a column in the dataframe'.format(column)

            for original, substitution in substitutions:
                # since the substitutions are implemented by a for loop over the elements in
                # `substitutions`, column, [(1,2),(2,4)] will be executed in sequence, leading to
                # 1 -> 4 and 2 -> 4.
                # TODO: correct this issue.
                df[column] = df[column].apply(lambda x: substitution if x == original else x)

        return df

    @staticmethod
    def drop_rows_with(df, value_specs):
        """
        Drop all rows where a given column has a particular value.
        :param df: Pandas dataframe
        :param value_specs: list of tuples with entries of the form
            (column, [value1, value2, ...])
        where column is the name of the dataframe column and values in the list are those to be
        discarded.
        :return: reduced dataframe
        """

        for column, values in value_specs:
            assert column in df.columns, '{} is not a column in the dataframe'.format(column)

            for value in values:
                df = df.drop(df[df[column] == value].index)

        return df

    @staticmethod
    def drop_columns(df, columns):
        """
        Drop columns
        :param df: Pandas dataframe
        :param columns: list with columns to drow
        :return: reduced dataframe
        """
        for column in columns:
            assert column in df.columns, '{} is not a column in the dataframe.'. format(column)
        return df.drop(columns=columns)

    @staticmethod
    def append_columns(df, column_specs):
        """
        Append columns, each with a specifiec default value. This can be useful when concatenating
        dataframes with non-identical indices.
        :param df: Pandas dataframe
        :param column_specs: list of tuples with entries
            (column, default_value)
        where column is the name of the dataframe column name and default_value is the value over
        the column. The column names must be different to existing ones.
        :return: extended dataframe
        """
        for (column, value) in column_specs:
            assert column not in df.columns, 'A column with the name {} already exists.'.format(column)
            df[column] = value
        return df

    @staticmethod
    def rename_columns(df, rename_specs):
        """
        Rename columns according to specification. We default the pandas method argument index
        with str.
        :param df: Pandas dataframe
        :param rename_specs: list of tuples of the form
            (original, final)
        where original and final are the original and final column names, respectively.
        :return: Pandas dataframe
        """
        return df.rename(index=str, columns={original: final for original, final in rename_specs})

    def execute_pipeline(self, df):
        """

        :param df:
        :return:
        """
        for task in self.pipeline:
            args = self.pipeline[task]
            if args is not None:
                df = getattr(self, task)(df, args)
            else:
                df = getattr(self, task)(df)

        return df


class PreprocessInteractionData(PreprocessGenericData):
    def __init__(self, pipeline):
        super().__init__(pipeline)

    _extended_dtypes = ['event_category']

    @staticmethod
    def _event_dtype():
        """
        Auxiliary method for creating a categorical data type for the 'type' data. We take into
        consideration that there exists an ordering in the existence of events given by
            broadcast_init -> start_session -> sent -> opened, session -> used, feedback,
        that is, in the relation A -> B, the existence of the event of type B is conditioned by
        the previous existence of an event A.
        :return: ordered Pandas CategoricalDtype object
        """
        return CategoricalDtype(
            categories=[
                'broadcast_init',
                'start_session',
                'sent',
                'opened',
                'session',
                'used',
                'feedback'
            ],
            ordered=True)

    @classmethod
    def set_edtypes(cls, df, edtype_specs):
        """
        Transforms dataframe columns into their corresponding extended types. This is necessary
        for custom dtypes, like categorical data with ordering.
        :param df: Pandas dataframe
        :param edtype_specs: list of tuples with entries
            (column, edtype)
        where column is a string specifying the name of a dataframe column and  edtype is another
        string specifying the expected type the elements in the column should have. Supported
        edtype strings are given in the class attribute _extended_dtypes.

        :return: transformed dataframe.
        """
        for (column, dtype) in edtype_specs:
            assert column in df.columns, '{} is not a column in the dataframe'.format(column)
            assert dtype in cls._extended_dtypes, '{} is not an (extended) supported type for conversion'.format(dtype)

            if dtype == 'event_category':
                try:
                    df[column] = df[column].astype(cls._event_dtype(), errors='ignore')
                except ValueError as ve:
                    print(ve)
        return df

    @staticmethod
    def for_user(df, idx):
        """
        Yield user dataframe for the User class from the final general dataframe.
        :param df: Pandas dataframe.
        :param idx: integer for user_id.
        :return: reduced Pandas dataframe.
        """
        return df[df['user_id'] == idx].reset_index(drop=True).drop(columns=['user_id'])

    @staticmethod
    def for_event(df, idx):
        """
        Yield event dataframe for the Event class from the final general dataframe.
        :param df: Pandas dataframe.
        :param idx: string for event type.
        :return: reduced Pandas dataframe.
        """
        df = df[df['type'] == idx].reset_index(drop=True)
        return PreprocessInteractionData._reduce_event_df(df, idx)

    @staticmethod
    def for_post(df, idx):
        """
        Yield post dataframe for the Post class from the final general dataframe.
        :param df: Pandas dataframe
        :param idx: integer for post_id
        :return: reduced Pandas dataframe
        """
        return df[df['post_id'] == idx].reset_index(drop=True).drop(columns=['post_id'])

    @staticmethod
    def _reduce_event_df(df, event_type):
        """
        Helper method for dropping specific columns for the event dataframes depending on the
        type of event.
        :param df: df already filtered by user from the general dataframe.
        :param event_type: string specifying event type
        :return:
        """
        # `broadcast_init` should have columns: [type, created_at]
        if event_type == 'broadcast_init':
            return df.drop(columns=['minutes', 'review']).reset_index(drop=True)
        # `start_session` should have columns: [type, created_at]
        elif event_type == 'start_session':
            return df.drop(columns=['minutes', 'review']).reset_index(drop=True)
        # `sent` should have columns: [type, created_at, post_id]
        elif event_type == 'sent':
            return df.drop(columns=['minutes', 'review']).reset_index(drop=True)
        # `opened` should have columns: [type, created_at, post_id]
        elif event_type == 'opened':
            return df.drop(columns=['minutes', 'review']).reset_index(drop=True)
        # `session` should have columns: [type, created_at, post_id, minutes]
        elif event_type == 'session':
            return df.drop(columns=['review']).reset_index(drop=True)
        # `used` should have columns: [type, index, created_at, post_id]
        elif event_type == 'used':
            return df.drop(columns=['minutes', 'review']).reset_index(drop=True)
        # `feedback` should have columns: [type, index, created_at, post_id, review]
        elif event_type == 'feedback':
            return df.drop(columns=['minutes']).reset_index(drop=True)
        # any other event type should have columns: [type, index, created_at, post_id, minutes, review]
        # since existing events have been exhausted, this should return an empty df.
        else:
            return df.reset_index(drop=True)


class SetUpDataframes(object):
    interaction_pipeline = {
        'rename_columns': [('value', 'minutes')],
        'replace_in_column': [('type', [('sended', 'sent'), ('Start_Session', 'start_session')])],
        'drop_rows_with': [('user_id', [0, 4, 5, 1311, 4865]), ('type', ['retested'])],
        'set_dtypes': [('created_at', 'datetime64')],
        'set_edtypes': [('type', 'event_category')],
    }

    feedback_pipeline = {
        'drop_rows_with': [('user_id', [0, 4, 5, 1311, 4865])],
        'rename_columns': [('value', 'review')],
        'append_columns': [('type', 'feedback')],
        'set_dtypes': [('created_at', 'datetime64'), ('review', 'int64')],
        'set_edtypes': [('type', 'event_category')]
    }

    content_pipeline = {
        'rename_columns': [('id', 'post_id')]
    }

    response_pipeline = {
        'set_dtypes': [('response', 'int64')],
        'drop_rows_with': [('response', [0])]
    }

    @classmethod
    def interaction_df(cls, raw_df):
        """
        Setup the interaction dataframe.
        :param raw_df: Raw dataframe directly read from the MySQL database by the interaction_data
        method in the ReadDatabase class.
        :return: interaction pandas dataframe ready for use.
        """
        preprocessor = PreprocessInteractionData(cls.interaction_pipeline)
        return raw_df.pipe(preprocessor.execute_pipeline)

    @classmethod
    def feedback_df(cls, raw_df):
        """
        Setup the feedback dataframe.
        :param raw_df: Raw dataframe directly read from the MySQL database by the feedback_data
        method in the ReadDatabase class.
        :return: feedback pandas dataframe ready for use.
        """
        preprocessor = PreprocessInteractionData(cls.feedback_pipeline)
        return raw_df.pipe(preprocessor.execute_pipeline)

    @classmethod
    def response_df(cls, raw_df):
        """
        Setup the responses dataframe.
        :param raw_df: Raw dataframe directly read from the MySQL database by the feedback_data
        method in the ReadDatabase class.
        :return: feedback pandas dataframe ready for use.
        """
        preprocessor = PreprocessInteractionData(cls.response_pipeline)
        return raw_df.pipe(preprocessor.execute_pipeline)


    @classmethod
    def overall_df(cls, raw_interaction_df, raw_feedback_df):
        feedback_df = cls.feedback_df(raw_feedback_df)
        interaction_df = cls.interaction_df(raw_interaction_df)
        df = pd.concat([interaction_df, feedback_df], sort=False, ignore_index=True) \
            .sort_values('created_at').reset_index(drop=True)
        df = PreprocessInteractionData.set_dtypes(df, [('review', 'int64')])
        return df

    @classmethod
    def content_df(cls, raw_df):
        """
        Setup the content dataframe.
        :param raw_df: Raw dataframe directly read from the MySQL database by the content_data
        method in the ReadDatabase class.
        :return: content pandas dataframe ready for use.
        """
        preprocessor = PreprocessInteractionData(cls.content_pipeline)
        return raw_df.pipe(preprocessor.execute_pipeline)

    @classmethod
    def feedback_matrix(cls, raw_df):
        """
        Produce feedback matrix with rows associated to posts, columns associated to users and entries
        given by the feedback given by a user to a post.
        :param raw_df: raw feedback dataframe.
        :return: pandas dataframe.
        """
        return cls.feedback_df(raw_df).pivot(index='post_id', columns='user_id', values='review')

    @classmethod
    def response_matrix(cls, raw_df):
        """
        Produce feedback matrix with rows associated to posts, columns associated to users and entries
        given by the feedback given by a user to a post.
        :param raw_df: raw feedback dataframe.
        :return: pandas dataframe.
        """
        return cls.response_df(raw_df).pivot_table(
            index='question_id', columns='user_id', values='response', aggfunc='mean')

