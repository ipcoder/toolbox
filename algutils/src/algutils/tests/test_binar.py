from ..binary import *


def test_set_bits():
    assert set_bits(0, 3, 4, 2) == 0x30
    assert set_bits(0xffff, 0, 4, 4) == 0xff0f


def test_get_bits():
    assert get_bits(0xffff, 0, 4) == 0xf
    assert get_bits(0xfff0, 3, 4) == 0xe


def test_extract_bits():
    assert extract_bits(0b01101001001, 0, 3) == 0b1001
    assert extract_bits(0b01101001001, 0, 0) == 0b1
    assert extract_bits(0b01101001001, 6, 9) == 0b1101

    assert (extract_bits(np.array([0b01011, 0b11001]), 3, 4) == np.array([0b01, 0b11])).all()


def test_split_bits():
    assert split_bits(0b01101001001, [2, 3]) == [0b01, 0b010]

    inp = np.array([0b01011, 0b11001])   # now lets take 2, 3 bits from each
    out = [np.array([0b11, 0b01]), np.array([0b010, 0b110])]
    assert all((xa == ya).all() for xa, ya in zip(split_bits(inp, [2, 3]), out))


def test_join_bits():
    a = np.random.randint(0, 4, size=(3, 4))
    b = np.random.randint(0, 8, size=(3, 4))

    joined = join_bits([2, 3], [a, b])
    ra, rb = split_bits(joined, [2, 3])

    assert (a == ra).all() and (b == rb).all()


def test_encode_array():
    print('\n................ encode_array ...................')
    encode_array(4, [11, 111, 227, 25], show=False)


def test_bin():
    print('\n................  bin * bin ................ ')
    b1 = Bin(10, 4)
    b2 = Bin(120, 12)

    b3 = b1 * b2
    print('%s + %s = %s' % (b1, b2, b3))

    assert isinstance(b3, Bin)
    assert bits_num(b3) == b1.bits + b2.bits
    assert int(b3) == int(b1) * int(b2)


def test_bin_div():
    print('\n................  bin // bin  ................ ')
    b1 = Bin(17, 5)
    b2 = Bin(120, 8)

    b3 = b2 // b1
    print('%s // %s = %s' % (b2, b1, b3))

    assert isinstance(b3, Bin)
    assert bits_num(b3) == b2.bits
    assert int(b3) == int(b2) // int(b1)


def test_num_bin_div():
    print('\n................  num // bin  ................ ')
    b1 = Bin(17, 5)
    b2 = 120

    b3 = b2 // b1
    print('%s // %s = %s' % (b2, b1, b3))

    assert isinstance(b3, Bin)
    assert bits_num(b3) == bits_num(120)
    assert b3.bits == 7     # bits_num(b3)
    assert int(b3) == int(b2) // int(b1)

    b2 = Bin(b2)
    b3 = b2 // b1
    print('%s // %s = %s' % (b2, b1, b3))

    assert b3.bits is None
    assert isinstance(b3, Bin)
    assert int(b3) == int(b2) // int(b1)
