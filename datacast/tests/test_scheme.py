import os
from pathlib import Path

import pytest

from toolbox.datacast.caster import PickleRelativePath
from algutils.param import TBox
from algutils.cache import Pickle, CacheInvalidError, CachedPipe, CacheMode
from algutils.paths import TransPath


def test_labels_extraction(tiny_stereo):
    for labels in tiny_stereo.caster.iter():
        assert isinstance(labels, dict) and len(labels) > 0


def test_pather_merger():
    NM = r'\w+?'
    pat = r'(?P<kind>(frames_cleanpass)|(disparity))/{subset}' \
          r'/{scene_1}/{scene_2}/{side}' \
          r'/{scene_3}\.(?P<ext>(?(3)pfm|png))'
    rgx = fr'(?P<kind>(frames_cleanpass)|(disparity))/(?P<subset>{NM})' \
          fr'/(?P<scene_1>{NM})/(?P<scene_2>{NM})/(?P<side>{NM})' \
          fr'/(?P<scene_3>{NM})\.(?P<ext>(?(3)pfm|png))'
    fmt = '{kind}/{subset}/{scene_1}/{scene_2}/{side}/{scene_3}.{ext}'

    pb = TransPath(mapper={}, pattern=pat)
    assert pb.regex.regex.pattern == rgx
    assert pb.form.str == fmt


def test_pather_regex_forms():
    pattern = r"{dataset}?/{*:inp}/{kind}{*:out}?" \
              r"(_{alg}(_(?P<ver>[a-z0-9\\.]+))?(_(?P<cfg>[a-z0-9]+))?)?" \
              r"{_*}\.tif"
    pather = TransPath(pattern)

    pattern2 = r"{dataset}?/{*:inp}/{kind}{*:out}?" \
               r"(_{alg}" \
               r"(_{ver:[a-z0-9\\.]+})?" \
               r"(_{cfg:[a-z0-9]+})?" \
               r")?" \
               r"{_*}\.tif"

    pather2 = TransPath(pattern2)
    assert pather2.regex.regex.pattern == pather.regex.regex.pattern
    assert pather2.form.str == pather.form.str


def test_pather_anonym():
    pattern = r"{dataset}?/{*:inp}/{kind}{*:out}?" \
              r"(_{alg}(_(?P<ver>[a-z0-9\\.]+))?(_(?P<cfg>[a-z0-9]+))?)?" \
              r"{_*}\.tif"
    pather = TransPath(pattern)

    named_labels = dict(dataset='MID', kind='disp', alg='PDS', ver='0', cfg='abcd')
    more_labels = {'nn': 'Nothing'}
    inp = dict(x=10, y=20)
    out = dict(ww='HI')

    path_out = pather(more_labels, **named_labels, inp=inp, out=out)
    path = 'MID/[x=10,y=20]/disp[ww=HI]_PDS_0_abcd_[nn=Nothing].tif'
    assert path_out == path

    labels = pather.regex.parse(path)
    ref_labels = named_labels | more_labels | inp | out
    ref_labels = {k: str(v) for k, v in ref_labels.items()}
    assert labels == ref_labels
    assert labels['nn'] == more_labels['nn']


def test_pather_anonym_optional():
    pattern = r"{*:inp}/{kind}{_*(provided):out}?" \
              r"_{alg}{_*suffix}?\.tif"
    pather = TransPath(pattern)

    path = pather(inp={'yes': 2}, kind='disp', alg='PDS')

    path = '[x=10,y=20]/disp_[ww=HI]provided_PDS_0_abcd_[dd=2]suffix.tif'
    labels = pather.regex.parse(path)
    assert labels == {
        'x': '10',
        'y': '20',
        'ww': 'HI',
        'dd': '2',
        'kind': 'disp',
        'alg': 'PDS_0_abcd'
    }

    path_no_inp = '[x=10,y=20]/disp_PDS.tif'
    labels = pather.regex.parse(path_no_inp)
    assert labels == {'x': '10', 'y': '20', 'kind': 'disp', 'alg': 'PDS'}

    path_no_inp = '[]/disp_PDS.tif'
    labels = pather.regex.parse(path_no_inp)
    assert labels is None  # inp is required anonymous group!


test_yaml_scheme = r"""
        pattern: 'data/level_{level}_{lid}/file_scene{scene}_id{sid}.txt'
        mappings:
            sid: {'0':'L', '1':'R'}
            lid: {'1':'easy', '2':'hard'}
        filters: "sid=='L' or level=='two'"
    """
test_items_num = 6  # 8 (content of the `data` folder in the scheme) - 2 filtered


@pytest.fixture
def pths_gen(tmp_path):
    file_path = Path('file.pkl')
    file_path_altered = Path(f'{tmp_path}/file_altered.pkl')
    # to use later as value of key 'path' in box
    correct_path = Path(f'{tmp_path}/{file_path}')
    # path manipulation
    splitted = tmp_path.__str__().split(os.sep)
    splitted[1] = 'tmpp'
    altered = os.sep.join(splitted)
    altered_path = Path(f'{altered}/{file_path}')
    return TBox({
        'file_path': file_path,
        'tmp_path': tmp_path,
        'correct_path': correct_path,
        'file_path_altered': file_path_altered,
        'altered_path': altered_path
    })


def test_relative_pickle_serializer(pths_gen):
    """
    Tests the serializer behaviour.
    The serializer has two modes:
        - save : Serialize
        - laod : Deserialize
    Saving with non-existing path as value of {path:value} dict FAILS
    Saving with non-existing path as serializer save 'file' argument FAILS
    Loading wrongly saved cache with this serializer may FAIL.
    """
    prp = PickleRelativePath(root=pths_gen.tmp_path, safe=True)
    # box init
    box = [TBox(path=pths_gen.correct_path)]
    altered_box = [TBox(path=pths_gen.altered_path)]
    # save with altered box - with wrong absolute path using _PickleRelativePath SHOULD FAIL
    with pytest.raises(ValueError):
        prp.save(pths_gen.correct_path, altered_box)
    # save with wrong absolute path using _PickleRelativePath SHOULD FAIL
    with pytest.raises(FileNotFoundError):
        prp.save(pths_gen.altered_path, box)
    # save with other serializer and loading with _PickleRelativePath may lead to CacheInvalidError
    Pickle.save(pths_gen.file_path_altered, [TBox(path=pths_gen.altered_path.__str__())])
    with pytest.raises(CacheInvalidError):
        prp.load(pths_gen.file_path_altered)
    # saving and loading with correct values SHOULD PASS
    prp.save(pths_gen.correct_path, box) and prp.load(pths_gen.correct_path)
    assert prp.add_root(pths_gen.file_path) == pths_gen.correct_path.__str__()


def test_piped_relative_pickle_serializer_(pths_gen):
    """
    Tests relative pickle behaviour when utilizes threw the caching mechanism.
    Mimics actual usage of the serializer in the platform to prompt the user on unintentionally breaking Cache internals.
    """
    prp = PickleRelativePath(root=pths_gen.tmp_path, safe=True)
    Pickle.save(pths_gen.file_path_altered, [TBox(path=pths_gen.altered_path.__str__())])
    # manipulating path to simulate cache-friendly naming
    pipe_friendly_path = pths_gen.file_path_altered.__str__().replace('file_altered.pkl', '_file_altered.pkl')
    # changing the file to allow load using CachedPipe stage.
    os.rename(pths_gen.file_path_altered.__str__(), pipe_friendly_path)

    # initializing stages
    stages = [CachedPipe.Source(os.listdir(pths_gen.tmp_path.__str__()), 'file_altered', serial=prp),
              CachedPipe.Map(lambda x: x, 'file_altered', serial=prp, copy=dict.copy),
              CachedPipe.Filter(lambda x: x, 'file_altered', serial=prp, copy=dict.copy)]

    # CachedPipe with KEEP mode SHOULD LOAD if cache exists
    pipe = CachedPipe(stages, folder=pths_gen.tmp_path / 'cache', mode=CacheMode.KEEP)

    # this SHOULD raise CacheInvalid for each and every stage
    for _, stage in enumerate(pipe.stages_iter(reverse=True)):
        with pytest.raises(CacheInvalidError):
            stage.serial.load(Path(pipe_friendly_path))


if __name__ == '__main__':
    # test_pather_merger()
    # test_format_to_regex_to_format()
    # test_relative_pickle_serializer(Path('/tmp/'))
    # test_piped_relative_pickle_serializer_(Path('/tmp/'))
    test_pather_anonym()
    test_pather_anonym_optional()
    pass
