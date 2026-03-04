from copy import copy

import box
import pytest

from algutils.param import TBox


def test_construct():
    b = TBox({
        "x.y.z": 10,
        "a": {"b": {"c": {}}}},
        y={"z": 6})
    assert b.x.y.z == 10
    assert b.a.b.c == {}
    assert b.y.z == 6


def test_set_get():
    b = TBox()
    b["x"] = 10
    assert b.x == 10
    b["y.z"] = 20
    assert b.y.z == 20
    b["y"] = 30  # override b.y.z
    assert b.y == 30

    with pytest.raises(AttributeError):
        b["y.z"] = 30
    with pytest.raises(AttributeError):
        b.w
    with pytest.raises(AttributeError):
        b.x.w


def test_access():
    tb = TBox()
    tb["a.b.c"] = 7
    # Check different access options
    assert tb.a.b.c == 7
    assert tb.a["b.c"] == 7
    assert tb.a.b["c"] == 7
    assert tb["a"].b.c == 7


def test_keys_items():
    # check whether keys method in default works as dict keys method
    tb = TBox(x={'y': [6, 7], 'z': "str"}, j=[1, 2, 3], i=1)
    k = tb.keys()
    assert k == {'x', 'j', 'i'}
    deep_k = tb.x.keys()
    assert deep_k == {'y', 'z'}

    # deep keys
    full_k = tb.keys(True)
    assert len(full_k) == 4
    assert tb.keys(True) == sorted({'x.y', 'x.z', 'j', 'i'})  # Keys in Box is sorted

    # full items
    assert tb.items(True) == sorted([('x.y', [6, 7]), ('x.z', 'str'), ('j', [1, 2, 3]), ('i', 1)])


def test_copy():
    tb = TBox(x={'y': [1, 2, 3]}, j=20, z="Str")
    c_tb = copy(tb)  # Shallow copy
    deepc_tb = tb.copy()  # Deep copy
    tb.x.y.append(12)
    assert tb != deepc_tb
    assert tb == c_tb
    tb['x'].y = [1, 2, 3, 4, 5]
    assert tb != deepc_tb
    assert tb != c_tb


def test_default_box():
    b = TBox()
    b = TBox(x=10, y=20)
    b.z = 10

    with pytest.raises(box.BoxKeyError):
        _ = b.f


def test_setdefault():
    b = TBox(x=10, y={'p': 20, 'r': {}})
    d = b.copy()

    d.n = 30
    b.setdefault('', default=d)
    assert b.n == d.n
    # deep setdefault
    b = TBox(x=10, y={'p': 20, 'r': {'t': {}}})
    b.setdefault("y", default=13)
    assert b.y == {'p': 20, 'r': {'t': {}}}

    b_2 = TBox(x=10, y={'p': 20, 'r': {'t': {}}})
    b.setdefault("y.p", default=13)
    assert b_2 == b
    b.setdefault("y.r", default=50)
    assert b_2 == b
    b.setdefault("y.w.q", default=50)
    assert b.y.w.q == 50

    # error for trying to use int as a dict
    with pytest.raises(KeyError):
        b.setdefault("y.p.q", default=50)

    # default is dict. not deep
    b.setdefault("w", default={"j": {"i": 6}})
    assert b.w == {"j": {"i": 6}}
    # default is dict. deep
    b.setdefault("a.b", default={"j": {"i": 6}})
    assert b.a.b == {"j": {"i": 6}}

    b = TBox(x=10, y={'p': 20, 'r': {'t': {}}})
    b_2 = TBox(x=10, y={'p': 20, 'h': 1, 'z': {}, 'r': {'t': {}}})
    # fill missing in TBox
    b.setdefault('', default=b_2)
    assert b == b_2


def test_find_key():
    b = TBox(default_box=True)

    b.one.three = 13
    b.three.four = 34
    b.one.five = 15
    b.one.six = 16
    b.two.two = 22

    assert b.find_key('four') == 'three.four'
    assert b.find_key('four', multi=True) == ['three.four']
    assert len(b.find_key('one', multi=True)) == 3
    assert b.find_key('two') == 'two.two'

    assert len(b.find_key('three', multi=True)) == 2
    start = b.find_key('three', part='start', multi=True)
    end = b.find_key('three', part='end', multi=True)
    assert len(start) == 1
    assert len(end) == 1
    assert start[0] != end[0]


def test_remove():
    b = TBox({
        "x.m.a": 1, "x.m.b": 2,
        "y.n.o": 33, "y.n.p": 44, "y.k.r": 55
    })

    assert b.remove('x.m') == TBox(y=b.y, x={})
    assert b.remove('x') == TBox(y=b.y)

    c = b.copy()
    c.discard('y')
    assert c == TBox(x=c.x)

    c = b.copy()
    c.discard(['y.n.o', 'x.m.b', 'y.k'])
    assert c == TBox({"x.m.a": 1, "y.n.p": 44})

    wrong_key = 'x.m.n'
    with pytest.raises(KeyError):
        assert wrong_key not in b
        b.remove(wrong_key)

    assert b.remove([wrong_key, 'y'], strict=False) == TBox(x=b.x)

    with pytest.raises(KeyError):
        b.discard([wrong_key, 'y', 'x.m.a'])

    b.discard([wrong_key, 'y', 'x.m.a'], strict=False)
    assert b == TBox({'x.m.b': 2})


if __name__ == '__main__':
    test_construct()
    test_set_get()
    test_access()
    test_keys_items()
    test_copy()
    test_default_box()
    test_setdefault()
    test_find_key()
    test_remove()