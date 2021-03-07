import os
import glob
import random
from collections import defaultdict
import re


class WorkLoad(object):
    r""" query workload from path """

    def __init__(self, workload_path):
        r"""
            Args:
                workload_path: sql directory
         """
        self.workload_path = workload_path
        self.sql_files = glob.glob(f"{self.workload_path}/*.sql")
        # unvalid = ['/home/ehds/data/join-order-benchmark-master/10c.sql',
    #            '/home/ehds/data/join-order-benchmark-master/10b.sql']

    def sample(self, num=1):
        assert num > 0
        sql_files = random.choices(self.sql_files, k=num)
        return [WorkLoad._get_sql_from_file(f) for f in sql_files]

    def get_query(self, file_name):
        full_path = os.path.join(self.workload_path, file_name)
        return WorkLoad._get_sql_from_file(full_path)

    def get_all_query(self):
        return [WorkLoad._get_sql_from_file(f) for f in self.sql_files]

    def get_all_query_names(self):
        """ get all query name ordered by id 
            Return:
                data:dict id->[...]
        """
        sql_files = self.sql_files
        sql_names = defaultdict(list)
        for sql_file in sql_files:
            name = os.path.split(sql_file)[1]
            template_id = re.search('\d+', name).group(0)
            sql_names[template_id].append(name)
        return sql_names

    def __len__(self):
        if self.sql_files == None:
            return 0
        return len(self.sql_files)

    @staticmethod
    def _get_sql_from_file(file_path, encoding='utf-8'):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        else:
            return None
