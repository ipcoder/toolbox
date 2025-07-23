from toolbox.utils.short import as_list, as_iter


def test_as_list():
    assert as_list('hi') == ['hi']
    assert as_list('hi', collect=set) == {'hi'}
    assert as_list(None) == []
    assert as_list(None, empty_none=False) == [None, ]
    assert as_list(a := [1, 2, 3]) is a
    assert as_list(a, collect=tuple) == tuple(a)
    assert as_list(a, collect=tuple, no_iter=list) == tuple([a])


def test_as_iter():
    assert list(as_iter('hi')) == ['hi']
    assert list(as_iter(None)) == []
    assert list(as_iter(None, empty_none=False)) == [None, ]
    assert as_iter(a := [1, 2, 3]) is a
