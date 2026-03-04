from ..datatools import *


def test_common_dict():
    dicts = dict(
        d1={1: 2, 3: 4, 5: 6, 7: 8},
        d2={1: 2, 5: 60, 7: 8},
        d3={1: 2, 7: 8},
        d4={1: 2, 3: 40, 7: 8},
    )

    assert common_dict(dicts) == {1: 2, 7: 8}
    assert common_dict([*dicts.values()], True) == ({1: 2, 7: 8}, [{3: 4, 5: 6}, {5: 60}, {}, {3: 40}])
