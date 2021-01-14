# -*- coding: UTF-8 -*-

import json
import re


def extract_object_from_str(data):
    """extract key value from data and conver to json format """
    data.replace(' ', '')

    def tostr(data):
        data = data.group(0)
        return data[0]+"\""+data[1:-1]+"\""+data[-1]

    data = re.sub(r',(.*?):', tostr, data)
    data = re.sub(r':([^\{]*?)[^\}],', tostr, data)
    data = re.sub(r'{(.*?):', tostr, data)
    data = re.sub(r':([^\"]*?)}', tostr, data)

    return json.loads(data)


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
