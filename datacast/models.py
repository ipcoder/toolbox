""" See ``./docs/data_resources.md`` """
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union, Literal, Iterable

from pydantic import DirectoryPath, root_validator, validator, Field, BaseModel, ValidationError

from toolbox.datacast.scan import GuideScan
from inu.env import EnvLoc
from toolbox.resman import resource as rc
from toolbox.resman.resource import ResModelError
from algutils import logger, as_list, Strings

_log = logger('datacast')
_datasource_yml = 'datasource.yml'

__all__ = ['DataSourceRM', 'SchemeRM', 'DatasetRM', 'CollectionRM', 'GuideScan', 'discover']


def _safe_update_kwargs(dct, **kws):
    for key, val in kws.items():
        if key in dct:
            raise ValueError(f"Assumed positional argument {key} already exists in kwargs")
        dct[key] = val


def _name_from(obj):
    """Get name from obj name, name-key or name-attr or return None"""
    if obj:
        if name := isinstance(obj, dict) and obj.get('name', None):
            return name
        if name := getattr(obj, 'name', None):
            return name
    return None


class SrcNotFoundError(ValueError):
    def __init__(self, *args, **kwargs):
        pass


@rc.locatable(folders=EnvLoc.DATASETS,
              skip_after_match=True, skip_under_match=True,
              max_files=20, max_folders=40)
class DataSourceRM(rc.ResourceModel, patterns=['.src.yml', _datasource_yml],
                   hash_exclude=['root']):
    """
    A model that holds descriptive information about a datasource as well as its location.
    """
    layout: Optional[str]
    root: DirectoryPath | str

    description: Optional[str]
    alias: Optional[Union[str, list[str]]]
    domain: Optional[Strings]
    realism: Optional[Literal["low", "moderate", "good", "high", "excellent", "true"]]
    origin: Optional[str]
    synthetic: Optional[bool]

    @validator('realism', pre=True)
    def real(cls, v):
        if v is True:
            v = 'true'
        return v

    @root_validator(pre=True)
    def from_file_info(cls, values):
        name, root, file = (values.get(_, None) for _ in ['name', 'root', cls._file_info_attr])
        # file = some/dataset_name/datasource.yml => (name = dataset_name, root = some/dataset_name)
        if file:  # try extracting `name` and `root` values from file path or name
            file = rc._FileInfo(**file) if isinstance(file, dict) else file
            if file.path.name == _datasource_yml:  # 1. from its parent folder if datasource.yml
                # ALLOW both root and file on copy construction
                if root and root != file.folder:
                    raise ValueError(f"Can't create root from different {_datasource_yml} and {root}")
                name = name or file.folder.stem
                root = file.folder
            else:  # 2. from its name
                name = name or file.base_name
                root = root or file.folder
        if not root:
            raise SrcNotFoundError(f"For {cls._kind_name} with {name=} root can't be found by "
                                      f"searching '{_datasource_yml}' under\n\t\t{cls._manager._locs}")

        values.update(name=name, root=root)
        return values

    @validator('alias')
    def aliases(cls, v):
        return as_list(v)

    @validator('layout', always=True, pre=True)
    def layout_from_name(cls, v, values):
        return v or values.get('name', None)

    def __init__(self, root_or_name: Path | str = None, /, **kws):
        """Support DataSourceModel(name | path, ...)"""
        if root_or_name is not None:
            arg = 'root' if rc.represents_path(root_or_name) else 'name'
            _safe_update_kwargs(kws, **{arg: root_or_name})
        super().__init__(**kws)


@rc.locatable(folders=EnvLoc.RESOURCES / 'schemes')
class SchemeRM(rc.ResourceModel, extra='allow',
               desc='Data Layout Scheme', patterns='.scm.yml',
               hash_exclude=['description', 'layout', 'bundle']):
    """
    A model for configuring PathScheme.
    The configuration must contain a pattern.

    The root that binds the scheme to the file-system in order to traverse it is determined on two ways:
     1. explicitly setting root
     2. explicitly passing the codename of the datasource's layout

    That yields different possibilities for scheme configuration:
    ::
        1. Full
        name: FT3D
        layout: FT3D
        search:
            pattern: "..."

        2. Minimal
        pattern:  "..."

    A full YAML configuration example is shown below:
    ::
        name: FT3D
        description: Flying Things 3D
        bundle: [ subset, scene ]
        mappings:
            subset: str.lower
            kind: { frames_cleanpass: image, disparity: disp }
            view: { left: L, right: R }

        if kind == 'image':
            alg: cam
            range: "0,255"

        if kind == 'disp':
           alg: GT

        search:
            pattern: "(?P<kind>(frames_cleanpass)|(disparity))/{subset}\
                      /{scene_1}/{scene_2}/{view}/{scene_3}\
                      \\.(?P<ext>(?(3)pfm|png))"
            samples:
              - disparity/TRAIN/A/0749/right/0015.pfm
              - frames_cleanpass/TEST/C/0145/right/0013.png

            ignore_case: True
            under_found: True
            max_depth: 3

    The values passed to _samples, if so exists, are being parsed with the given pattern.
    An error with them is raised on fail.

    """
    # descriptive
    description: Optional[str] = None
    layout: str = None
    bundle: Optional[Strings] = None  # Consider: Move to Dataset?

    search: GuideScan
    # labeling
    mappings: dict = Field(default_factory=dict)
    labels: dict = Field(default_factory=dict)
    reverse: bool = Field(False, description="Labels may be reversed to create paths")

    @validator('name', always=True, pre=True)
    def name_from_file_name(cls, name, values):
        return name or (file := values.get(cls._file_info_attr)) and file.base_name

    @validator('layout', always=True, pre=True)
    def layout_from_name(cls, v, values):
        return v or values.get('name', None)

    @validator('search', always=True, pre=True)
    def search_from_pattern(cls, search):
        def _check(_search):
            if not re.search(r'[\\*+{}?<>]', _search):
                raise ResModelError(f'Given {_search} is not a valid regex pattern in {cls.__name__}')
            return True

        if isinstance(search, str):
            _check(search)
            search = GuideScan(pattern=search)
        elif isinstance(search, dict):
            search_pattern = search.get('pattern', None)
            if not search_pattern:
                raise ResModelError(f'Given {search} does not have pattern key {cls.__name__}')
            _check(search_pattern)
            search = GuideScan(**search)
        return search

    @root_validator(pre=True)
    def sugar_extra_labels(cls, values):
        """Move undefined fields under labels field"""
        assert (labels_attr := 'labels') in cls.__fields__, "must be defined by the model!"
        labels = values.setdefault(labels_attr, {})
        d = {f: v for f, v in values.items() if f not in cls.__fields__}
        labels.update(d)
        for k in d: del values[k]  # delete floating labels
        return values

    def __init__(self, pattern_or_name: str | GuideScan = None, /, **kws):
        """
        Allows optioinal special initialization:
        ::
            SchemeRM('/some/{pattrern}') == SchemeRM(search=GuideScan(pattern='/some/{pattern}'))
        """
        if pattern_or_name is not None:
            if isinstance(pattern_or_name, str) and re.search(r'[\\*+{}?<>]', pattern_or_name):
                _safe_update_kwargs(kws, search=GuideScan(pattern=pattern_or_name))
            else:
                _safe_update_kwargs(kws, name=pattern_or_name)
        super().__init__(**kws)


class Selection:

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    def validate(cls, v):
        if isinstance(v, str):
            if re.fullmatch(r'\d+(\.\d*)?%', v):
                v = float(v[:-1]) / 100
            elif re.fullmatch('\([\w, +-]+\)', v):
                v = eval('slice' + v)
            else:
                ValueError(f'Invalid sample selection value: "{v}"')
        elif isinstance(v, tuple):
            v = slice(*v)
        elif isinstance(v, list) and not all(isinstance(x, int) for x in v):
            raise ValueError(f"List of indices in sample {v} must contain only int")
        elif isinstance(v, int):
            assert v >= 0, "Sample size may not be negative"
        elif not isinstance(v, (float, slice)):
            raise ValueError(f"Invalid value for samples selection: {v}")

        if isinstance(v, float):
            assert 0 <= v <= 1, "Fraction of samples must be in [0,1] range"

        return v


class DSample(BaseModel):
    """
    Support full sample format with multiple forms of selection:
    ::
        sample:
            selection: None | 25 | .4 | '40%' | '(10, 20)' | '(1000,  None, 4)'
            shuffle: True

    And also shortcut version with selection value:
    ::
        sample:  selection: None | 25 | .4 | '40%' | (10, 20) | (1000,  None, 4)

    """
    selection: Selection
    shuffle: bool = False
    groups: bool | str | list[str] = None


@rc.locatable(folders=EnvLoc.RESOURCES / 'datasets')
class DatasetRM(rc.ResourceModel, patterns='.ds.yml', refers=[SchemeRM, DataSourceRM]):
    scheme: SchemeRM
    source: DataSourceRM
    filters: Union[str, dict, None] = None
    transforms: dict[str, Union[dict, None]] = None
    sample: DSample | Selection = None
    labels: dict = None

    @classmethod
    def _find_scheme_by_layout(cls, src_layout: str | None):
        """ Find all schemes with same layout as source_layout.
        return: MASTER scheme (scheme.layout == scheme.name)
                OR THE ONLY scheme with source_layout
                OR raise ResFieldNotExists
                """
        if not src_layout:
            raise rc.ResNoFieldError(f"Can't guess scheme for {cls._kind_name} from\n{cls.source=}")
        found = []
        master_found = 0
        for scheme_cfg in SchemeRM._manager.list('cfg'):
            if src_layout == scheme_cfg.get('layout', None):
                found.append(scheme_cfg)
                if scheme_cfg.get('name') == src_layout:  # master scheme
                    master_found += 1

        if len(found) == 0:
            raise rc.ResModelError(f"Can't find schemes with layout:{src_layout} for {cls._kind_name}")
        if master_found != 1:
            raise rc.ResModelError(
                f"{master_found} schemes found with {src_layout} as layout, only MASTER layout supported")

        return found.pop()

    @root_validator(pre=True)
    def fill_resources(cls, values):
        """Fills mandatory fields: name, scheme, source"""
        fields = ['name', 'scheme', 'source', cls._file_info_attr]
        name, scheme, source, file_info = (values.get(x, None) for x in fields)

        if isinstance(scheme, str):  # otherwise: dict, model, None
            scheme = SchemeRM(scheme)

        try:  # try building the source
            # allow defining Dataset by providing source as path and scheme as pattern or name
            if isinstance(source, (Path, str)):  # otherwise: dict, model, None
                source = DataSourceRM(source)  # source is root or name

            # name => file.name => source.name => scheme.name => None
            values['name'] = name = (name or
                                     file_info and file_info.base_name
                                     or _name_from(source)
                                     or _name_from(scheme))

            # source => scheme name => name => Datasource(source)
            if isinstance(source, dict):
                source = DataSourceRM(**source)
            elif not isinstance(source, DataSourceRM):  # source is None or unsupported type
                source = source or _name_from(scheme) or name
                source = DataSourceRM(source)
        except SrcNotFoundError as ex:
            raise ValueError(f"{cls._kind_name} failed to initialize from {source=}:\n\t {ex}")

        values['source'] = source

        # scheme => SchemeModel(name) => (scheme.layout is source.layout)
        if not scheme:
            if scheme := SchemeRM.find_resource(name):
                values['scheme'] = scheme
            else:  # try to determine scheme by layout if it is mentioned in the source
                values['scheme'] = cls._find_scheme_by_layout(source.layout)
        else:
            values['scheme'] = scheme

        return values

    @validator('sample', pre=True)
    def validate_sample(cls, sample, values):
        """ Allow `sample: num | slice`"""
        if not (sample is None or isinstance(sample, (dict, DSample))):
            sample = DSample(selection=sample)
        return sample

    @validator('sample', always=True)
    def sample_bundle(cls, sample, values):
        """If bundle not provided use that of scheme"""
        if sample is not None and sample.groups is None:
            bundle = values['scheme'].bundle
            if bundle is not None:
                sample.groups = as_list(bundle)
        return sample


@rc.locatable(folders=EnvLoc.RESOURCES / 'collections')
class CollectionRM(rc.ResourceModel, desc="Collection of Datasets", patterns='.col.yml', refers=DatasetRM):
    datasets: list[DatasetRM]
    label_datasets: bool | None = None
    query: Optional[str | dict] = None
    bundle: Optional[list[str]] = None
    description: Optional[str] = None

    @root_validator(pre=True)
    def dataset_from_name(cls, values):
        values.setdefault('datasets', values.get('name', None))
        return values

    @validator('name', pre=True, always=True)
    def fill_name(cls, name, values):
        if not name and (file := values[cls._file_info_attr]):
            name = file.base_name
        return name

    _DST = str | DatasetRM | dict

    @validator('datasets', pre=True, always=True)
    def check_datasets(cls, datasets: _DST | list[_DST], values):
        if not datasets:
            if isinstance(datasets, Iterable) or not (datasets := values.get('name', None)):
                raise rc.ResModelError(f"{cls.__name__} initialized with empty collection of datasets")

        # FixMe: Inherited Dataset (which can have changed params) has same name!
        return [DatasetRM.from_config(cfg) for cfg in as_list(datasets)]

    @root_validator
    def set_datasets_labels(cls, values):
        """If `label_datasets` add 'dataset' label to every Dataset"""
        if values.get('label_datasets', None) and values.get('datasets', None):
            for ds in values['datasets']:
                ds.labels = (ds.labels or {}) | {'dataset': ds.name}
        return values

    def __init__(self, /, *datasets_or_name: str | DatasetRM | Iterable[str | DatasetRM], **kws):
        """
        Adding support for:
        ::
            CollectionRM('ColName')
            CollectionRM('Ds1', 'Ds2', DatasetRM('DS3'))
            CollectionRM(['Ds1', 'Ds2', DatasetRM('DS3')])

        :param datasets_or_name:
        :param kws:
        """
        if (args_num := len(datasets_or_name)) > 1:
            _safe_update_kwargs(kws, datasets=list(datasets_or_name))
        elif args_num == 1:
            if isinstance(v := datasets_or_name[0], list):
                _safe_update_kwargs(kws, datasets=list(v))
            else:
                _safe_update_kwargs(kws, name=v)
        super().__init__(**kws)


def discover(*models, **kwargs):
    from toolbox.resman import resman
    models = models or ('datasource', 'dataset', 'scheme', 'collection')
    return resman.discover(*models, **kwargs)
