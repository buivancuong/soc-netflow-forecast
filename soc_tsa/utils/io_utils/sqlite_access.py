import sqlite3
import os
import logging

# BUILD SQLite3 database file
# def create_sqlite_db(db_name):
#     os.system("sqlite3 " + db_name)


class SQLiteDB:

    def __init__(self, db_path):
        self.db_path = db_path

    def create_db_connection(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
        except:
            logging.error('Cannot connect to SQLite DB')


# BUILD models table on SQLite3 db
# def create_table(connection, table_name):
#     cursor = connection.cursor()
#     custom_table = """CREATE TABLE models (
#         model_id Varchar,
#         model_path Varchar,
#         sensor Varchar,
#         ts_field Varchar
#     )"""
#     cursor.execute(custom_table)
#     connection.close()

    def add_model(self, table_name, model_id, model_path, sensor, ts_field):
        if self.connection is not None:
            cursor = self.connection.cursor()
            query_string = 'INSERT INTO ' + table_name + \
                ' (model_id, model_path, sensor, ts_field) VALUES (' + model_id + \
                ', ' + model_path + ', ' + sensor + ', ' + ts_field + ' );'
            try:
                cursor.execute(query_string)
            except:
                logging.error(
                    'Cannot execute the SQLite query to add new model.')
            self.connection.commit()
            self.connection.close()
        else:
            logging.info("Cannot add new model because cannot connect to DB")

    def delete_model(self, table_name, model_id):
        if self.connection is not None:
            cursor = self.connection.cursor()
            query_string = 'DELETE FROM ' + table_name + ' WHERE model_id = ' + model_id
            try:
                cursor.execute(query_string)
            except:
                logging.error(
                    "Cannot execute the SQLite query to delete the model")
            self.connection.commit()
            self.connection.close()
        else:
            logging.warning(
                "Cannot delete the model because cannot connect to DB")

    def list_models(self, table_name):
        if self.connection is not None:
            cursor = self.connection.cursor()
            query_string = 'SELECT model_id FROM ' + table_name
            result = list()
            try:
                for record in cursor.execute(query_string):
                    result.append(record)
            except:
                logging.error(
                    "Cannot execute the SQLite query to list all models")
            self.connection.commit()
            self.connection.close()
            return result
        else:
            logging.warning("Cannot list models because cannot connect to DB")
            return None
