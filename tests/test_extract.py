from utils.extract import extract_object_from_str


def test_extract_object_from_str():
    data = "{time:69.4µs, loops:2, cop_task: {num: 1, max: 268.3µs, proc_keys: 0, rpc_num: 1, rpc_time: 209.7µs, copr_cache_hit_ratio: 0.00}}"
    extract = extract_object_from_str(data)
    assert type(extract) == dict

