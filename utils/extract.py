# -*- coding: UTF-8 -*-

import json
import re


def extract_object_from_str2(data):
    """extract key value from data and conver to json format """
    data.replace(' ', '')

    def tostr(data):
        data = data.group(0)
        return data[0]+"\""+data[1:-1]+"\""+data[-1]

    def erase_unused_common(data):
        data = data.group(0)
        return '}'
    data = re.sub(r',(.*?):', tostr, data)
    data = re.sub(r':([^\{]*?)[^\}],', tostr, data)
    data = re.sub(r'{(.*?):', tostr, data)
    data = re.sub(r':([^\"]*?)}', tostr, data)
    data = re.sub(r'(,})', erase_unused_common, data)

    return json.loads(data)

def extract_object_from_str(data):
    data = data.replace(' ', '')
    execution_time_match = re.search(r'time:(.*?)[,}$]', data)
    execution_time = "0s"
    if execution_time_match != None:
        execution_time = execution_time_match.group(1)
    return json.loads(f'{{"time":"{execution_time}"}}')
def convert_analyze_to_object(data):
    """ make analyze info str to json format """
    if "AnalyzeInfo" in data:
        ana = "{"+data["AnalyzeInfo"]+"}"
        ana = ana.replace(' ', '')
        data["AnalyzeInfo"] = extract_object_from_str(ana)
    if "children" in data and data["children"] != None:
        for i in range(len(data["children"])):
            data["children"][i] = convert_analyze_to_object(
                data["children"][i])
    return data
