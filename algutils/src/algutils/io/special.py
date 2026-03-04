from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union, Optional, Sequence

import numpy as np

from .. import as_list

Keys = Optional[Union[Sequence[str], str]]


def savep(file, v):
    """ Save pickle variable
    """
    with open(file, "wb") as f:
        pickle.dump(v, f)


def loadp(file):
    """ Load saved pickle
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def middlebury_calib(source: Union[str, 'Path']):
    """
    Read Middlebury calibration data format
    :param source: path to the calib file (usually */calib.txt)
    :return: StereoCam TBox object with calibration parameters
    """
    from .camera import StereoCam, TBox

    source = Path(source) if isinstance(source, str) else source
    c = TBox.from_yaml(source.read_text('utf-8').replace('=', ': '))

    for key in ['cam0', 'cam1']:
        #   f  0  cx
        #   0  f  cy
        #   0  0   1
        c[key] = np.array([*map(str.split, c[key][0].split(';'))], dtype=float)

    return StereoCam(resolution=(c.width, c.height),
                     baseline=c.baseline,
                     units='mm',
                     center=c.cam0[:2, 2],
                     focal=c.cam0[0, 0])


def extract_attr(meta: dict | str, nested=False, required: Keys = None, skip: Keys = None,
                 search_nodes=('meta', None, 'attributes', 'attr')) -> dict[str, ...]:
    """
    Extract attributes from the file's meta-data.

    :param meta: meta-data (dict) or file (name or path object)
    :param required: if defined extract only specified key(s) from the meta node
    :param skip: ignore specified key(s) if found in the meta node
    :param nested: allow to extract dict items
    :param search_nodes: nodes names in hierarchical formats to search for meta-data
    :return: dict with extracted meta-data {key: value}
    """

    def is_accepted(kv):
        k, v = kv
        return k not in skip \
            and (not required or k in required) \
            and (nested or not isinstance(v, dict))

    def meta_nodes_candidates(data):
        meta_keys = set(as_list(search_nodes, empty_none=False))
        check_top = None in meta_keys  # search as top keys if meta node not found
        meta_keys.discard(None)

        if meta_keys:
            for key, val in data.items():
                if key in meta_keys and isinstance(val, dict):
                    yield val
        if check_top:
            yield data

    if not isinstance(meta, dict):
        from . import read_meta  # FixMe: old refactoring error?
        meta = read_meta(meta)

    skip = skip and set(skip) or {}
    required = required and set(required)

    for node in meta_nodes_candidates(meta):
        res = dict(filter(is_accepted, node.items()))
        if res: return res  # return first found node content
