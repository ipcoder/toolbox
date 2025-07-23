import pickle

import numpy as np

from geom.shapes import Vec2d, Rect


def test_creation_and_access():
    v = Vec2d(111, 222)
    assert v.x == 111 and v.y == 222
    v.x = 333
    v[1] = 444
    assert v[0] == 333 and v[1] == 444


def test_math():
    v = Vec2d(111, 222)
    assert v + 1 == Vec2d(112, 223)
    assert v - 2 == [109, 220]
    assert v * 3 == (333, 666)
    assert v / 2.0 == Vec2d(55.5, 111)
    assert v / 2 == (55.5, 111)
    assert v ** Vec2d(2, 3) == [12321, 10941048]
    assert v + [-11, 78] == Vec2d(100, 300)
    assert v / [10, 2] == [11.1, 111]


def test_reverse_math():
    v = Vec2d(111, 222)
    assert 1 + v == Vec2d(112, 223)
    assert 2 - v == [-109, -220]
    assert 3 * v == (333, 666)
    assert [222, 888] / v == [2, 4]
    assert [111, 222] ** Vec2d(2, 3) == [12321, 10941048]
    assert [-11, 78] + v == Vec2d(100, 300)


def test_unary():
    v = Vec2d(111, 222)
    v = -v
    assert v == [-111, -222]
    v = abs(v)
    assert v == [111, 222]


def test_length():
    v = Vec2d(3, 4)
    assert v.length == 5
    assert v.length_sqr() == 25
    assert v.normalize_return_length() == 5
    assert v.length == 1
    v.length = 5
    assert v == Vec2d(3, 4)
    v2 = Vec2d(10, -2)
    assert v.distance(v2) == (v - v2).length


def test_angles():
    v = Vec2d(0, 3)
    assert v.angle == 90
    v2 = Vec2d(v)
    v.rotate(-90)
    assert v.angle_between(v2) == 90
    v2.angle -= 90
    assert v.length == v2.length
    assert v2.angle == 0
    assert v2 == [3, 0]
    assert (v - v2).length < .00001
    assert v.length == v2.length
    v2.rotate(300)
    np.isclose(v.angle_between(v2), -60)
    v2.rotate(v2.angle_between(v))
    np.isclose(v.angle_between(v2), 0)


def test_high_level():
    basis0 = Vec2d(5.0, 0)
    basis1 = Vec2d(0, .5)
    v = Vec2d(10, 1)
    assert v.convert_to_basis(basis0, basis1) == [2, 2]
    assert v.projection(basis0) == (10, 0)
    assert basis0.dot(basis1) == 0


def test_cross():
    lhs = Vec2d(1, .5)
    rhs = Vec2d(4, 6)
    assert lhs.cross(rhs) == 4


def test_comparison():
    int_vec = Vec2d(3, -2)
    flt_vec = Vec2d(3.0, -2.0)
    zero_vec = Vec2d(0, 0)
    assert int_vec == flt_vec
    assert int_vec != zero_vec
    assert (flt_vec == zero_vec) == False
    assert (flt_vec != int_vec) == False
    assert int_vec == (3, -2)
    assert int_vec != [0, 0]
    assert int_vec != 5
    assert int_vec != [3, -2, -5]


def test_inplace():
    inplace_vec = Vec2d(5, 13)
    inplace_ref = inplace_vec
    inplace_src = Vec2d(inplace_vec)
    inplace_vec *= .5
    inplace_vec += .5
    inplace_vec /= (3, 6)
    inplace_vec += Vec2d(-1, -1)
    assert inplace_vec == inplace_ref


def test_pickle():
    testvec = Vec2d(5, .3)
    testvec_str = pickle.dumps(testvec)
    loaded_vec = pickle.loads(testvec_str)
    assert testvec == loaded_vec


def test_rect():
    p1 = Vec2d(10, 20)
    p2 = Vec2d(20, 30)

    r = Rect(p1, p2)
    assert r == r
    print(r, Rect(p1, dim=p2 - p1))
    assert r == Rect(p1, dim=p2 - p1)

    assert (15, 25) in r
