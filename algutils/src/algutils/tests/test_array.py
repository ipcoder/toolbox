from ..array import *
from ..array import _wC2G
import pytest
import numpy as np


pytest.skip("Array tests require refactoring", allow_module_level=True)

@pytest.fixture(scope="module", autouse=True)
def registered_forms():
    DForm._parse_name('R3:4N.2b18')

    DForm('ImgNet', dtype='u1', color=Color.RGB,
          avr=np.array([0.485, 0.456, 0.406]) * 255,
          std=np.array([0.229, 0.224, 0.225]) * 255)


def test_form_array_naming():
    assert all(DForm('ImgNet').avr == np.array([0.485, 0.456, 0.406]) * 255)
    x = str(DForm('Bu1'))

    df_cnu1 = DForm(color='RGB', std=1, avr=0, dtype=np.uint8)
    assert DForm('CNu1') == df_cnu1
    assert not DForm('CNu1').same(df_cnu1)

    assert DForm('N2.5|-3f2') == DForm(std=2.5, avr=-3, dtype=np.float16)
    assert DForm('R1N|1.2f4') == DForm(min=0, max=1, avr=1.2, dtype=np.float32)
    assert DForm('Gb10') == DForm(color=Color.GRAY, min=0, max=1023, dtype=np.uint16)
    assert DForm('s2') == DForm(min=-1, max=1, dtype='i1')
    assert DForm('Bu1') == DForm(color=Color.BIN, dtype='i1')

    for df in DForm.list_registered(out=True).values():
        if df.norm_dim < 2:
            assert df._auto_name() == df.name


def test_form_array_class():
    a = [[.1, -10, 20]]

    df = 'Gb10'
    A1 = FormArray(df)
    A2 = FormArray(DForm(df))
    A3 = FormArray('Array', dform=df)
    assert A1 == A2 == A3

    # check if unregisterd names with autoname off  raise error
    DForm.auto_naming(False)
    with pytest.raises(TypeError):
        A1 = FormArray('Cu2')

    # works because interpreted as uint16 by dtype
    A2 = FormArray('u2')

    # create from name as type
    assert FormArray('uint8') == FormArray(DForm('u1'))

    DForm.auto_naming(True)

    # todo add test for change of bases and attrs


def test_array():
    assert Array(10).shape == ()
    assert Array([]).shape == (0,)
    assert Array([11]).shape == (1,)
    assert Array([11, 22]).shape == (2,)
    assert Array([[11, 22, 33]] * 4).shape == (4, 3)

    print(Array([[11, 22, 33]] * 4))

    n = 20
    a = Array(np.random.rand(n, n))
    assert isinstance(a, Array)
    assert isinstance(a, np.ndarray)
    assert isinstance(a[:1, :1], Array)
    assert isinstance(a.view(int), Array)

    assert type(a) == Array
    assert a.dform == Array.dform
    assert a.dtype == np.dtype(np.float32)  # default dtype of Array class

    form = DForm(dtype=np.int8, min=10, max=20)
    a = Array(np.random.rand(n, n), form)
    assert type(a) != Array
    assert a.dform == form
    assert a.dtype == form.dtype
    assert a.dtype == np.int8


def test_instantiate_array():
    a = [1, 2, 3]
    assert isinstance(Array(a, dform='f4'), Array)
    assert isinstance(Array(a, dform='Nu1'), FormArray('Nu1'))

    b = FormArray.from_array(a)
    assert b.dform == DForm(int)

    b = FormArray.from_array([-89, .1, 1999.23], dform='i8', name='MyArray', min=1, max=2)
    assert b.dform == DForm(dtype='i8', name='MyArray', min=1, max=2)
    print(b)


def test_array_repr():
    n = 20
    a = Array(np.random.rand(n, n))

    assert str(a).count(str(n)) == 2 and str(a).count('\n') == 0

    with pytest.raises(KeyError):
        Array.print_options(wrong=10)

    assert str(Array(range(4))).count('\n') == 0
    item = '_0.123'
    type(a).print_options(
        rows=n + 1, cols=n + 1,
        floatmode='fixed',
        precision=len(item.split('.')[1]),
        linewidth=(n + 1) * (len(item))
    )
    assert str(a).count('\n') >= n

    type(a).print_options(info=False)
    assert str(a[:2, :3]).count('\n') == 1

    type(a).print_options(info=True)
    assert str(a[:2, :3]).count('\n') == 2


def test_init_data_form():
    df = DForm()
    assert not df.extra_info()

    df = DForm(dtype=float)
    assert not df.extra_info()
    assert df.dtype == np.float_

    with pytest.raises(NameError):
        DForm('ok')

    df = DForm('Gb10')
    assert df.color is Color.GRAY

    df = DForm('Cu1')
    assert df.color.channels == 3


def test_data_form_transform():
    DF = cu8 = DForm('Cu1')
    a_c8 = Array(np.random.randint(DF.min, DF.max, (4, 6, 3), dtype=DF.dtype), DF)
    assert a_c8.dform.color.channels == 3
    print(f"{a_c8=}")

    # todo move to test_transform_color
    DF = DForm('Gb10')
    a_g10 = Array(np.random.randint(DF.min, DF.max, (4, 6), dtype=DF.dtype), DF)
    t_c8 = a_g10.transform(cu8)
    print(f"{t_c8=}")
    assert t_c8.dform == cu8
    assert (t_c8[:, :, 1] == np.rint(a_g10 * (255 / 1023)).ndarray.astype('uint8')).all()

    a_imn = Array(a_c8, DForm('ImgNet'))
    t_n01 = a_imn.transform(DForm('CNf4'))

    print(t_n01)


def test_transform_norm():
    FormArray.global_print_options(norm=True)
    # todo transform of Array lacks the do_norm and do_range parameters. Include them there
    # transform of avr and std with size ( 1 - > 1 )
    dform1 = DForm(color="RGB", std=10, avr=3, dtype=np.float16)
    dform2 = DForm(color="GRAY", std=5, avr=1, dtype=np.float16)
    dform3 = DForm(color="RGB", std=5, avr=1, dtype=np.float16)

    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    c = a.transform(dform3)

    std_scale = b.dform.std / a.dform.std  # the same for b and c, same norms
    # check if transform of norms done properly regardless of color
    assert (b == Array(np.dot(a.ndarray, _wC2G).squeeze() * std_scale + b.dform.avr - a.dform.avr * std_scale,
                       'f2')).all()
    assert (c == (a * std_scale + b.dform.avr - a.dform.avr * std_scale).transform('f2')).all()

    # transform of avr and std with size ( 1 - > 3 )
    dform2 = DForm(color="RGB", std=[5, 6, 7], avr=[1, 2, 3], dtype=np.float32)
    b = a.transform(dform2)
    std_scale = b.dform.std / a.dform.std  # vector
    # check if transform of norms done properly for each channel
    assert ((b[:, :, ch] == a[:, :, ch] * std_scale[ch] + b.dform.avr[ch] - a.dform.avr * std_scale[ch]).all()
            for ch in range(3))

    # transform of avr and std with size ( 3 - > 1 ) when RGB -> RGB
    dform1 = DForm(color="RGB", std=[5, 6, 7], avr=[1, 2, 3], dtype=np.float32)
    dform2 = DForm(color="RGB", std=5, avr=1, dtype=np.float32)
    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    std_scale = b.dform.std / a.dform.std  # vector
    # check if transform of norms done properly for each channel
    assert ((b[:, :, ch] == a[:, :, ch] * std_scale[ch] + b.dform.avr - a.dform.avr[ch] * std_scale[ch]).all()
            for ch in range(3))

    # transform of avr and std with size ( 3 - > 1 ) when RGB -> GRAY
    dform1 = DForm(color="RGB", std=[5, 6, 7], avr=[1, 2, 3], dtype=np.float32)
    dform2 = DForm(color="GRAY", std=5, avr=1, dtype=np.float32)
    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    a_weighted = np.dot(a, _wC2G).squeeze()
    avr_weighted = np.dot(np.resize(a.dform.avr, 3), _wC2G).squeeze()
    std_weighted = np.dot(np.resize(a.dform.std, 3), _wC2G).squeeze()
    std_scale = b.dform.std / std_weighted  # scalar
    # check if transform of norms done properly with the RGB to GRAY weights
    assert (b == (a_weighted * std_scale + b.dform.avr - avr_weighted * std_scale).transform('f4')).all()

    # transform of avr and std with size ( 3 - > 3 )
    dform1 = DForm(color="RGB", std=[5, 6, 7], avr=[1, 2, 3], dtype=np.float32)
    dform2 = DForm(color="RGB", std=[10, 11, 12], avr=[4, 5, 6], dtype=np.float32)
    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    std_scale = b.dform.std / a.dform.std  # vector
    # check if transform of norms done properly for each channel
    assert (
        (b[:, :, ch] == a[:, :, ch] * std_scale[ch] + b.dform.avr[ch] - a.dform.avr[ch] * std_scale[ch]).all()
        for ch in range(3))

    # if avr and std not provided, an error occur
    dform1 = DForm(dtype=np.float32)
    dform2 = DForm(std=5, avr=1, dtype=np.float32)
    a = form_array([1., 2.], dform=dform1)  # real avr is 1.5, real std is 0.5
    with pytest.raises(TypeError):  # FIXME change to specific error
        b = a.transform(dform2)

    # todo need to implement new interface for calculating norm
    # you can ask to calculate avr and std
    dform1 = DForm(dtype=np.float32)
    dform2 = DForm(std=5, avr=1, dtype=np.float32)
    a = form_array([1., 2.], dform=dform1)  # real avr is 1.5, real std is 0.5
    b = a.transform(dform2, norm="calculate")
    std_scale = dform2.std / a.std()
    assert (b == a * std_scale + dform2.avr - a.mean() * std_scale).all()


def test_transform_range():
    # transform of range
    dform1 = DForm(min=-1, max=3, dtype=np.float16)
    dform2 = DForm(min=-20, max=20, dtype=np.float16)
    a = form_array([1., 1.], dform=dform1)
    b = dform2.transform(a, dform1)
    range_scale = (dform2.max - dform2.min) / (dform1.max - dform1.min)
    # check if the range transformation is correct
    assert (b == a * range_scale + dform2.min - dform1.min * range_scale).all()


def test_transform_color():
    # changing cax to other axis
    dform1 = DForm(color="RGB", cax=-1, dtype=np.float16)
    dform2 = DForm(color="GRAY", cax=1, dtype=np.float16)
    dform3 = DForm(color="RGB", cax=1, dtype=np.float16)
    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    c = a.transform(dform3)
    # check if the cax is the same regardless of colors
    assert b.dform.cax == c.dform.cax == 2
    # check if the size of dim in the cax is correct
    assert b.shape[b.dform.cax] == 1
    assert c.shape[c.dform.cax] == 3

    # changing from none to cax
    dform1 = DForm(color="GRAY", dtype=np.float16)
    dform2 = DForm(color="GRAY", cax=-1, dtype=np.float16)
    dform3 = DForm(color="RGB", cax=-1, dtype=np.float16)
    a = form_array(np.random.rand(4, 6), dform=dform1)
    b = a.transform(dform2)
    c = a.transform(dform3)
    # check if the cax is the same regardless of colors
    assert b.dform.cax == c.dform.cax == -1
    # check if the size of dim in the cax is correct
    assert b.shape[b.dform.cax] == 1
    assert c.shape[c.dform.cax] == 3
    # check if all channels in RGB image are the same
    assert (c[:, :, 0] == c[:, :, 1] == c[:, :, 2]).all()

    # changing from cax to none
    dform1 = DForm(color="RGB", cax=-1, dtype=np.float16)
    dform2 = DForm(color="GRAY", dtype=np.float16)
    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    # check if the shape of b is correct
    assert b.shape == a.shape[:2]
    # check if Gray image created correctly
    assert b == np.dot(a, _wC2G).squeeze()

    # changing cax but it means the same
    dform1 = DForm(color="RGB", cax=-1, dtype=np.float16)
    dform2 = DForm(color="GRAY", cax=2, dtype=np.float16)

    a = form_array(np.random.rand(4, 6, 3), dform=dform1)
    b = a.transform(dform2)
    assert a == b
    assert a.dform.cax != b.dform.cax


def test_transform_type():
    # types transformation
    dform1 = DForm(avr=150, dtype=np.int16)
    dform2 = DForm(avr=40, dtype=np.int32)
    a = Array([100, 200], dform1)
    b = a.transform(dform2)
    c = a.transform(DForm('i1'))
    d = a.transform(DForm('f2'))
    assert b.dtype == b.dform.dtype == np.dtype('int32')
    assert c.dtype == c.dform.dtype == np.dtype('int8')
    assert d.dtype == d.dform.dtype == np.dtype('float16')

    # changing from RGB int to Gray float
    dform1 = DForm(color='RGB', dtype=np.uint8)
    dform2 = DForm(color='GRAY', dtype=np.float, min=0, max=1)
    a = form_array(np.random.randint(0, 255, (4, 6, 3)), dform=dform1)
    b = a.transform(dform2)
    assert b.dform.dtype == np.float
    assert np.min(b) >= b.dform.min
    assert np.max(b) <= b.dform.max


def test_transform_combined():
    dform1 = DForm(min=-1, max=3, dtype=np.float16)
    dform2 = DForm(std=4, avr=2, min=-1, max=3, dtype=np.float16)
    # Can't use range and norm at the same time
    with pytest.raises(ValueError):
        a = Array([1., 1.], dform1)
        b = dform2.transform(a, DForm(min=-20, max=20, std=5, avr=1, dtype=np.float16))

    # changing color and have norm info, with missing range in input
    with pytest.raises(TypeError):  # FixMe: what causes the error?
        dform1 = DForm(color='RGB', dtype=np.uint8, avr=[1, 2, 3], std=[2, 3, 4])
        a = form_array(np.random.randint(0, 255, (4, 6, 3)), dform=dform1)
        b = a.transform(dform2)

    # RGB -> RGB where norm collapses from 3 -> 1, with range, but it's the same
    dform1 = DForm(color='RGB', dtype=np.float, avr=[1, 2, 3], std=[2, 3, 4], min=0, max=1)
    dform2 = DForm(color='RGB', dtype=np.float, avr=2, std=3, min=0, max=1)
    a = form_array(np.random.randint(0, 255, (4, 6, 3)), dform=dform1)
    b = a.transform(dform2)
    assert len(b.shape) == 3
    assert len(b.dform.avr) == len(b.dform.std) == 1

    # todo transform of two or more characaristics at the same time
    pass


def test_same():
    # equal
    assert DForm('N2.5|-3f2') == DForm(std=2.5, avr=-3, dtype=np.float16)
    # but not the same
    assert not DForm('N2.5|-3f2').same(DForm(std=2.5, avr=-3, dtype=np.float16))
    # same because of same name and same parsing
    assert DForm('N2.5|-3f2').same(DForm(name='N2.5|-3f2'))
    # nor the same because of change in one attribute
    assert not DForm('N2.5|-3f2').same(DForm(name='N2.5|-3f4'))

    # todo add same method tests
    pass


def test_subclassing_array():
    # todo check functionality of new classes that are subclasses of Array
    pass


def test_dform():
    Array.dform = DForm('u1')
    a = Array([0, 0])
    assert (np.asarray(a) == np.zeros([2])).all()
    b = Array([0.7, 0.7])
    # checking if the array is changing because of using Array with dtype = int8
    assert (np.asarray(b) == np.ones([2])).all()


def test_asform():
    n = 20
    a = Array(np.random.rand(n, n))
    a.asform(DForm('u8'))


def test_dform_transform():
    dform1 = DForm(color="RGB", std=10, avr=3, dtype=np.float16)
    dform2 = DForm(color="GRAY", std=5, avr=1, dtype=np.float16)

    a = form_array(np.random.randint(0, 255, (4, 6, 3)), dform=dform1)
    # returns ndarray by default
    b = dform2.transform(a, dform1)
    assert type(b) is np.ndarray

    # using specific array for out
    dform1 = DForm(std=4, avr=2, color='RGB', dtype=np.float16)
    dform3 = DForm(std=4, avr=2, color='RGB', dtype=np.float32)
    a = form_array([[[1., 2., 3.]]], dform=dform1)
    c = form_array([[[0, 0, 0]]], dform=dform3)
    dform3.transform(a, dform1, out=c)
    assert (a == c).all()


def test_transform():
    n = 20
    a = Array(np.random.rand(n, n))
    b = a.transform(DForm('f8'))
    c = (a == b)
    print('c=', c)
    assert (a == b).all()

    b = a.transform('f4')
