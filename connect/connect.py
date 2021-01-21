# -*- coding: UTF-8 -*-

import MySQLdb
import json
import re
from utils.extract import convert_analyze_to_object


class DBConnect():
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

    def analyze(self, sql):
        analyze_sql = f"explain analyze {sql}"
        data = self.cursor.execute(analyze_sql)
        # Expected to get json format analyzeinfo
        data = self.cursor.fetchone()[0]
        # print(data)
        data = json.loads(data)

        return convert_analyze_to_object(data)
