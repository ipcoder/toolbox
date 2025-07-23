import pytest

from toolbox.utils.datatools import zip_dict
from toolbox.utils.wrap import name_func_outputs, namedtuple

dicts = {1: 11, 2: 12, 3: 13}, {1: 21, 2: 22, 4: 24}, {1: 31, 3: 33}


def test_zip_dict_strict():
    try:
        res = zip_dict(*dicts, strict=True)
    except KeyError:
        pass
    else:
        assert False  # Exception is expected for mismatched dicts in strict mode!


def test_zip_dict_fillvalue():
    fillvalue = 'XXX'
    assert zip_dict(*dicts, fillvalue=fillvalue)[4][::2] == (fillvalue,) * 2


def test_zip_dict_skip():
    assert len(zip_dict(*dicts, skip=True)) == 1


def test_namedtuple():
    vals = (1, 2, 3)
    fields = tuple('xyz')
    NT = namedtuple('Test', fields, vals)
    t = NT()
    assert t == vals
    assert t._fields == fields

    slc = slice(0, 2)
    assert t._part(slc) == t[slc]
    assert t._part(slc)._fields == fields[slc]


def test_name_outputs():
    f0 = lambda x: tuple(range(x)) if x > 1 else 0
    names = list('xyz')

    with pytest.raises(TypeError):
        f = name_func_outputs(f0, 'xyz')  # must be list of names

    f = name_func_outputs(f0, names)
    out = f(3)
    assert (out.x, out.y, out.z) == tuple(range(3))

    f2 = name_func_outputs(f, names)
    assert f is f2
    assert f2._name_outputs['nest'] == f._name_outputs['nest'] == 0

    with pytest.raises(RuntimeError):
        f2 = name_func_outputs(f, names, nest=False)

    f2 = name_func_outputs(f, names, nest=True)
    assert f2 is not f
    assert f2(3) == f(3) == f(3)
    assert f2._name_outputs['nest'] == 1

    f2 = name_func_outputs(f, names[:2])   # a different wrapping
    assert f is not f2
    assert f2._name_outputs['nest'] == 1

    with pytest.raises(RuntimeError):
        out = f(2)

    f = name_func_outputs(f0, names, adjust=True)
    out = f(2)
    assert (out.x, out.y) == tuple(range(2))

    f = name_func_outputs(f0, names, adjust=None)
    out = f(2)
    assert out == tuple(range(2)) and not hasattr(out, '_fields')
    out = f(1)


    # out dict
    f = name_func_outputs(f0, names, out_type=dict, adjust=True)
    out = f(2)
    assert (out['x'], out['y']) == (0, 1)
    assert f(1) == {'x': 0}

    f = name_func_outputs(f0, names, out_type=dict, adjust=None)
    out = f(2)
    assert out == tuple(range(2))
    assert f(1) == 0


if __name__ == '__main__':
    test_namedtuple()
