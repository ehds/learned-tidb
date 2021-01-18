from utils import join_order
import sys


def test_extract_join_conditions():
    data = "inner join, equal:[eq(test.b.id, test.a.id)]"
    info = join_order.extract_join_conditions(data)
    assert type(info) == list
    assert len(info) == 1


def test_extract_join_type():
    data = "inner join, equal:[eq(test.b.id, test.a.id)]"
    info = join_order.extract_join_type(data)
    assert type(info) == str
    assert info == 'inner join'


def test_extract_selection_info():
    data = "gt(cast(test.a.name), 2), like(test.a.name, \"%ab\", 92), lt(cast(test.a.name), 5), not(isnull(test.a.id))"
    info = join_order.extract_selection_info(data)
    assert type(info) == list
    assert len(info) == 4
    assert info[0].function == 'gt'
    assert info[1].function == 'like'
    assert info[2].function == 'lt'
    assert info[3].function == 'not'


def test_extract_join_tree():
    tree = join_order.extract_join_tree('data/test.json')
    assert type(tree) == join_order.JoinPlan


def test_convert_tree_to_trajectory():
    tree = join_order.extract_join_tree('data/test.json')
    trajectories = join_order.convert_tree_to_trajectory(tree)
    assert type(trajectories) == list
    assert len(trajectories) == 2
