from toolbox.utils.datatools import *


def test_common_dict():
    dicts = dict(
        d1={1: 2, 3: 4, 5: 6, 7: 8},
        d2={1: 2, 5: 60, 7: 8},
        d3={1: 2, 7: 8},
        d4={1: 2, 3: 40, 7: 8},
    )

    assert common_dict(dicts) == {1: 2, 7: 8}
    assert common_dict([*dicts.values()], True) == ({1: 2, 7: 8}, [{3: 4, 5: 6}, {5: 60}, {}, {3: 40}])


def test_rm_keys():
    dd = {x: x for x in ['one', 'two', '_three', '_four_', 'Yes']}
    assert set(rm_keys(dd, ['two', r'_\w+?_', 'Ye.*']).keys()) == {'one', '_three'}


def test_drop_undef():
    from toolbox.utils import drop_undef
    dct = dict(x=False, y=None, z=0)
    assert drop_undef(**dct) == {'x': False, 'z': 0}  # drop all None from keyword arguments
    assert drop_undef(ns=dct) == {'x': False, 'z': 0}  # drop all None from namespace dict
    assert drop_undef('x', 'y', ns=dct) == {'x': False}  # drop all except x, y, and them if None
    assert drop_undef(ns=dct, y=0) == {'x': False, 'y': None, 'z': 0}  # drop y if 0, others if None
    assert drop_undef('x', 'y', ns=dct, x=False, y=0) == {'y': None}  # drop z regardless
    assert drop_undef(ns=dct, x={False, 0}.__contains__) == {'z': 0}

    UNDEF = object()  # define particular UNDEF object to keep Nones
    assert drop_undef(_undef=UNDEF, x=10, y=None, z=UNDEF) == {'x': 10, 'y': None}


def test_filter():
    filters = dict(
        conditions='height * 2 < width - 10',
        name=['Sam', 'David'],
        age=int(18).__le__,
        side='right'
    )
    cond = Filter(filters)
    cond_keys = {'height', 'width', 'name', 'age', 'side'}
    labels = dict(height=10, width=4, name='Nick', age=100, side='top')
    assert cond_keys.issubset(labels.keys())
    assert cond(labels) is False

    from pandas import DataFrame
    df = DataFrame([
        dict(height=10, width=4, name='Nick', age=100, side='top'),
        dict(height=10, width=40, name='Sam', age=10, side='right'),
        dict(height=10, width=35, name='David', age=20, side='right')
    ])
    assert cond_keys.issubset(df.columns)
    assert len(row := df[df.apply(cond, axis=1)]) == 1 and row.index.item() == 2
