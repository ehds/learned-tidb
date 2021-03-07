# -*- coding: UTF-8 -*-

import MySQLdb
import json
import re
from utils.extract import convert_analyze_to_object
from utils.file_helper import write_db_info
from utils.join_order import convert_execute_time_to_ms
import math
import time


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

        self.db = MySQLdb.connect(host, user, passwd,
                                  database, port, charset)
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database
        self.port = port
        self.charset = charset
        self.cursor = self.db.cursor()
        self.tables = []
        self.columns = {}
        # get db info
        self._init_db_info()
        self.latency_record = {}

    def reconnect(self):
        self.db.close()
        self.db = MySQLdb.connect(self.host, self.user, self.passwd,
                                  self.database, self.port, self.charset)

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

    def explain(self, sql, analyze=False):
        sql_prefix = "explain " + ("analyze " if analyze else "")
        explain_sql = sql_prefix + sql
        try:
            data = self.cursor.execute(explain_sql)
            # Expected to get json format analyzeinfo
            data = self.cursor.fetchone()[0]
            data = json.loads(data)
            return convert_analyze_to_object(data)
        except Exception as e:
            print(e)
            if "gone away" in str(e):
                print("reconnect")
                self.reconnect()
            return None

    def get_est_rows(self, sql):
        explain_info = self.explain(sql)
        if explain_info == None:
            return 1e10
        # limit always at the top of explain info
        est_rows = explain_info['estRows']
        # act_rows = limit_info['ActRows']
        return float(est_rows)

    def get_latency2(self, sql, cache=False):
        if cache and (sql in self.latency_record):
            print("hit")
            return self.latency_record[sql]
        start = time.time()
        analyze_info = self.explain(sql, analyze=True)
        result = time.time()-start
        if cache:
            self.latency_record[sql] = result
        return result

    def get_latency(self, sql, cache=False):
        if cache and (sql in self.latency_record):
            return self.latency_record[sql]
        analyze_info = self.explain(sql, analyze=True)
        if analyze_info == None:
            return 1e10
        latency = convert_execute_time_to_ms(
            analyze_info['AnalyzeInfo']['time'])
        assert latency > 0
        if cache:
            self.latency_record[sql] = latency
        return float(latency)

    def analyze(self, sql):
        analyze_info = self.explain(sql, True)
        # Expected to get json format analyzeinfo
        if analyze_info != None:
            return analyze_info
        return None

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
