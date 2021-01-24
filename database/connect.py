# -*- coding: UTF-8 -*-

import MySQLdb
import json
import re
from utils.extract import convert_analyze_to_object
from utils.file_helper import write_db_info


class Column():
    def __init__(self, column_name):
        self.name = column_name
        self.cardinality = 0
        self.columns = []


class Table():
    def __init__(self, table_name):
        self.name = table_name
        self.columns = set()

    def add_column(self, column_name):
        self.columns.append(Column(column_name))


class DB():
    def __init__(self, host='localhost', user='root', passwd='', database='', port=3306, charset="utf8"):
        """
        args:
            host: remote host url
            user: username
            passwd: password
            database: database
            port: port default 3306
            charset: charset default utf-8
        """

        db = MySQLdb.connect(host, user, passwd,
                             database, port, charset)
        self.cursor = db.cursor()
        self.database = database
        self.tables = []
        self.columns = {}
        # get db info
        self._init_db_info()

    def _init_db_info(self):
        self.tables = self.get_all_tables()
        for table in self.tables:
            self.columns[table] = self.get_table_columns(table)
        # format db.table.column
        self.unique_columns = []
        for table in self.tables:
            for column in self.columns[table]:
                self.unique_columns.append(f'{self.database}.{table}.{column}')

        dbinfo = {}
        dbinfo["name"] = self.database
        dbinfo["tables"] = self.tables
        dbinfo["columns"] = self.columns
        dbinfo["flatten_columns"] = self.unique_columns
        write_db_info(dbinfo, self.database)

    def analyze(self, sql):
        analyze_sql = f"explain analyze {sql}"
        data = self.cursor.execute(analyze_sql)
        # Expected to get json format analyzeinfo
        data = self.cursor.fetchone()[0]
        # print(data)
        data = json.loads(data)

        return convert_analyze_to_object(data)

    def get_all_tables(self):
        query_sql = f"select table_name from information_schema.tables where table_schema='{self.database}'"
        data = self.cursor.execute(query_sql)
        data = self.cursor.fetchall()
        tables = [item[0] for item in data]
        return tables

    def get_table_columns(self, table_name):
        query_sql = f"select column_name from information_schema.columns where table_schema='{self.database}' and table_name='{table_name}'"
        data = self.cursor.execute(query_sql)
        data = self.cursor.fetchall()
        columns = [item[0] for item in data]
        return columns
