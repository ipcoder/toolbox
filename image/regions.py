from typing import Union, Optional

import numpy as np
import toolz as tz

from toolbox.utils import nptools as npt


class Regions(dict):
    """
    Collection labeling regions in 2D array.

    Organized as a dict of binary masks (of same shape) for every kind of regions.
    Support arithmetic on the regions as logical operation on the masks.

    Overrides iteration protocol to yield Regions object built around every region
    item in the container (instead of keys, as in dict)
    """

    def __init__(self, regions: Union[dict, np.array] = None,
                 rgn_codes: Optional[dict] = None, **kw_rgs):
        """
        Create from a dict of maps or a single map and dict of codes
        :param regions: dict of maps - then just built upon it, or
                        map of values - then dict of masks is formed using
                        `rgn_codes` dict mapping codes in the map into keys
        :param rgn_codes: mapping of names into values in `regions` array
        :param kw_rgs: regions maps as kw arguments
        """
        if rgn_codes:
            assert isinstance(regions, np.ndarray)
            regions = tz.valmap(regions.__eq__, rgn_codes)
        regions = {**(regions or {}), **kw_rgs}
        super().__init__(regions)

    def __and__(self, other):
        # print('====> [&]', list(self.keys()),
        #       list(other.keys()) if isinstance(other, dict) else '[M]')

        def _and(r1, m1, r2, m2):
            t1, t2 = m1.dtype.kind, m2.dtype.kind
            if t1 == 'b':
                if t2 == 'b':
                    return f"{r1} & {r2}", m1 & m2
                else:
                    return r1, npt.fill_mask(m2, ~m1)
            elif t2 == 'b':
                return r1, npt.fill_mask(m1, ~m2)
            return r1, m1

        if isinstance(other, dict):
            return Regions(dict(_and(*it1, *it2) for it1 in self.items()
                                for it2 in other.items()))
        else:
            return Regions({r: (m & other) for r, m in self.items()})

    def __iter__(self):
        return (Regions({k: self[k]}) for k in self.keys())

    def items(self):
        """Iterator over regions (name, data)"""
        return ((k, self[k]) for k in self.keys())

    def __or__(self, other):
        return Regions({f"{r1} | {r2}": m1 | m2
                        for r1, m1 in self.items() for r2, m2 in other.items()})

    def __xor__(self, other):
        return Regions({f"{r1} ^ {r2}": m1 ^ m2
                        for r1, m1 in self.items() for r2, m2 in other.items()})

    def __add__(self, other):
        return Regions({**self, **other})

    def __sub__(self, other):
        return

    def __repr__(self):
        if len(self):
            shapes = set(map(lambda x: x.shape, self.values()))
            types = set(map(lambda x: x.dtype, self.values()))
            if len(shapes) == 1 and len(types) == 1:
                h, w = shapes.pop()
                tp = types.pop()
                return ', '.join(self.keys()) + f": [{h}×{w}]{tp.kind}{tp.itemsize}"
        return "\n".join(f"{k}: [{m.shape[0]}×{m.shape[1]}]"
                         f"{m.dtype.kind}{m.dtype.itemsize}"
                         for k, m in self.items())

    def __str__(self):
        return self.__repr__()

    def __iadd__(self, other: 'Regions'):
        self.update(other)
        return self

    def __call__(self, *kinds):
        """Return subset of the regions as a new Regions object"""
        return Regions({kind: self[kind] for kind in kinds})

    def __invert__(self):
        return Regions({'~' + k: ~m if m.dtype is np.dtype(bool) else m
                        for k, m in self.items()})

    @property
    def key(self):
        """Assuming there is only one region returns its name, or None"""
        return next(iter(self.keys())) if len(self) == 1 else None

    @property
    def mask(self):
        """Assuming there is only one region returns its mask, or None"""
        return next(iter(self.values())) if len(self) == 1 else None

    @property
    def masks(self):
        return Regions({k: m for k, m in self.items() if m.dtype.kind == 'b'})

    @property
    def scores(self):
        return Regions({k: m for k, m in self.items() if m.dtype.kind != 'b'})

    @property
    def areas(self):
        return tz.valmap(np.sum, self)

    @property
    def area(self):
        return self.mask.sum() if len(self) == 1 else None

    @property
    def shape(self):
        """Return shape if all shapes are same or None"""
        shape = None
        for k, m in self.items():
            if shape is None:
                shape = m.shape
            elif m.shape != shape:
                return None
        return shape

    @staticmethod
    def from_larger_than(tag: str, score, thr, out):
        """
        Helper function to use when creating regions calculators
        :param tag: name of this regions kind
        :param score: score 2d array - base for the output
        :param thr: threshold to convert score into binary mask: score > thr
        :param out: format and content of the output:
                - '<tag>'  - binary mask score > thr or score < thr
                - 'score'  - float score (before threshold application)
                - 'region' - Regions obj with '<tag>' mask
                - 'all'    - Regions obj with '<tag>' mask and '<tag>_score'
                Use thr=None & 'region' for Regions obj with '<tag>_score'
        :return:
        """
        score_rgn = Regions({f'{tag}_score': score})
        if thr is None:
            if out in {'region', 'all'}:
                return score_rgn
            elif out == 'score':
                return score
            raise ValueError(f"Can't calculate {tag} mask without `thr`!")
        else:
            if out == 'score':
                return score

            with np.errstate(invalid='ignore'):
                mask = score > thr

            if out == tag:
                return mask
            if out == 'all':
                return Regions({tag: mask}) + score_rgn
            if out == 'region':
                return Regions({tag: mask})
