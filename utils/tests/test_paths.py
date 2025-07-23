import pytest
from toolbox.utils import drop_undef

pat = r"""
    {dataset}
    /{scene}
    (?:/{mode}{*:mode_par}?)?
    (?:/proc/{proc}{*:proc_par})?
    /{kind:image|disp}      # image: alg=cam, disp: alg=dpe
    {*:cam_par}?            # camera params
    (?:
      _{alg:[a-zA-Z0-9]+}
       (?:@{ver:[A-Za-z.0-9]+})?
       (?:\#{cfg:[^_]+})?
       {*:args}?
    )?
    _{view:L|R}
    (?:_{fid:\d+})?
    \.tif
    """


@pytest.fixture(scope='module')
def trans_path():
    from toolbox.utils.paths import TransPath
    tp = TransPath(pat)
    print(tp.regex.regex.pattern)
    return tp


def test_parse_by_pattern(trans_path):
    test_strings = [
        "minimal/small_cube/image[exp=20]_L.tif",
        "minimal/small_cube/disp[exp=20]_IDF@V2.2#[enc=mobile]_L.tif",
        "minimal/small_cube/image_L_019.tif",
        "minimal/small_cube/image[exp=20,gain=34]_L.tif",
        "minimal/small_cube/disp[exp=20,gain=2.5]_DPE@4.01#f45_L.tif",
        "minimal/small_cube/disp[gain=1.2]_IDF@S1.2#9f30_L.tif",
        "minimal/small_cube/active/disp[exp=20,gain=34]_IDF_L.tif",
        "minimal/small_cube/active[res=640x480]/proc/avr[n=10]/image_L.tif",
        "minimal/small_cube/active/proc/avr[n=100]/image[exp=20,gain=0.3]_L.tif",
        "minimal/small_cube/active/proc/hdr_avr[n=64,exps=3]/disp_DPE#fa67_L.tif"
    ]

    from toolbox.utils.strings import dict_str

    tp = trans_path

    for s in test_strings:
        print('=' * 80, s, '_' * 80, sep='\n')
        if labels := tp.regex.parse(s):
            print()
            print(dict_str(
                drop_undef(**labels),
                to=': ', sep='\n')
            )
        else:
            print('ERROR')


def test_format_by_pattern(trans_path):
    # -------  prepare labels
    cam_par = dict(exp=20, gain=1.2)
    mode = 'active'
    mode_par = dict(res='640x480')
    proc = 'avr'
    proc_par = dict(num=100)

    labels = dict(dataset='dataset', scene='scene')

    # ------ form items
    items = [
        imL := dict(kind='image', view='L'),
        imL | dict(cam_par=cam_par),
        imL_mode := imL | dict(mode=mode, mode_par=mode_par, fid='001'),
        imL_mode | (proc_lbs := dict(proc=proc, proc_par=proc_par)),
        dpe_R := dict(kind='disp', alg='DPE', ver=4100, cfg='face', view='R'),
        dpe_R | proc_lbs
    ]

    for item in items:
        item |= labels
        path = trans_path(**item)
        parsed = drop_undef(**trans_path.regex.parse(path))

        expected = {}   # flatten anonymous subgroups
        for k, v in item.items():
            expected.update(v if isinstance(v, dict) else {k: v})
        expected = {k: str(v) for k, v in expected.items()}  # parsed values are str!

        assert parsed == expected
