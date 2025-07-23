from __future__ import annotations

import re
from types import ModuleType
from typing import (Collection, TypeAlias, Callable, Type, TYPE_CHECKING, Union,
                    TypeVar, Generic, Literal, Optional, Any)

from pandas import DataFrame
from pydantic import FilePath, DirectoryPath, BaseModel, Extra, root_validator, ValidationError

import toolbox.param.models as pm

from toolbox.utils import logger, as_iter, drop_undef, as_list
from toolbox.utils.events import timed, time
from toolbox.utils.filesproc import PathT, Path, normalize, Locator, represents_path
from toolbox.utils.wrap import CaseInsEnum, enum_attr

if TYPE_CHECKING:
    from toolbox.datacast import DataCaster, DataCollection

_log = logger('resman')

__all__ = ['ModelsManager', 'ResManager', 'ResourceModel', 'find_resource', 'resman',
           'locatable', 'ResNameError', 'ResNoFieldError', 'ResNotFoundError']


class ResError(Exception):
    """Base Resource Error"""
    pass


class ResModelError(ResError):
    """Model Error"""
    pass


class ResNameError(ResError):
    """Unknown Resource Name"""
    pass


class ResNotFoundError(ResError):
    pass


class ResNoFieldError(ResError):
    pass


class _FileInfo(BaseModel):
    path: FilePath
    folder: DirectoryPath
    base_name: str
    suffix: str

    def __init__(self, path: PathT, folder=None, base_name=None, suffix=None):
        path = Path(path)

        sfx_sz = sum(map(len, path.suffixes))
        if not all([folder, base_name, suffix]):
            folder: Path = path.parent
            base_name = path.name[:-sfx_sz]
            suffix = path.name[-sfx_sz:]
        super().__init__(path=path, folder=folder, base_name=base_name, suffix=suffix)

    def __repr__(self):
        return f"FInf<{str(self.path)}>"


class ResourceModel(pm.YamlModel, extra=Extra.forbid):
    """
    A special subclass of ``YamlModel`` introducing the ``name`` field to subclssing models.

    That allows managing such models and their instances as `resources`.

    **Subclassing Arguments**

    ``ResourceModel`` accepts all the subclassing arguments necessary to deinfe its base class ``YamlModel``:

    >>> class MyMangedModel(ResourceModel, patterns='.calib.cfg', desc='Camera Calibration'):
    ...     ...
    >>> class FormatVedModel(ResourceModel, file_format: FileFormat = ...):
    ...     ...
    """

    _RT1: TypeAlias = Type["ResourceModel"]
    _RT = Union[_RT1, list[_RT1]]

    _file_info_attr = 'cfg_file_info'  # class attribute to use instead of string where _file_info is needed
    cfg_file_info: _FileInfo = None  # init in pre_root, excluded from to_yaml in __init_subclass__

    _manager: Optional[ResManager]
    _refers: Optional[_RT]

    name: Optional[str] = None

    @root_validator(pre=True)
    def set_file_cfg(cls, values):
        values.setdefault(cls._file_info_attr, None)
        return values

    def to_yaml(self, filename=None, **kws):
        """
        Generate a YAML representation of the model.

        *Always excludes `cfg_file_info` attribute from all the models tree.*

        Supports ``yaml_dump`` kws:
            include, exclude:
               Fields to include or exclude. See `dict()`.
            by_alias : bool
                   Whether to use aliases instead of declared names. Default is False.
            skip_defaults, exclude_unset, exclude_defaults, exclude_none
                Arguments as per `BaseModel.dict()`.
            sort_keys : bool
                If True, will sort the keys in alphanumeric order for dictionaries.
                Default is False, which will dump in the field definition order.
            default_flow_style : bool or None
                Whether to use the "flow" style in the dumper. By default, this is False,
                which uses the "block" style (probably the most familiar to users).
            default_style : {None, "", "'", '"', "|", ">"}
                This is the default style for quoting strings, used by `ruamel.yaml` dumper.
                Default is None, which varies the style based on line length.
            indent, encoding, kwargs
                Additional arguments for the dumper.

        :param filename: optional name to dump into, otherwise return the text buffer
        """
        with self.exclude_context(self._file_info_attr) as exclude:
            kws.update(exclude=exclude.union(  # add file_info_attr to the
                as_list(kws.get('exclude', None))  # excludes from the argument
            ))  # ONLY file_info_attr is excluded from all the sub-models!
            return super().to_yaml(filename=filename, **kws)

    def __init_subclass__(cls, *, refers: Optional[_RT] = None, **kwargs):
        """
        Optional arguments when subclassing `ResourceModel`:

        :param refers: other resource(s) this one refers to
        :param kwargs: kw arguments to `YamlModel` subclassing.
        :return:
        """
        cls._kind_name = re.sub(r'Model|RM|ResModel', '', cls.__name__)
        cls._refers = getattr(cls, '_refers', []) + as_list(refers, no_iter=type(BaseModel))

        kwargs['hash_exclude'] = as_list(kwargs.get('hash_exclude', [])) + [cls._file_info_attr]
        super().__init_subclass__(**kwargs)
        cls._manager = ModelsManager.register(cls)

    def _iter(
            self,
            to_dict: bool = False,
            by_alias: bool = False,
            include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']] = None,
            exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']] = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
    ):
        _exclude = {self._file_info_attr}

        if isinstance(exclude, dict):
            from pydantic.utils import ValueItems
            exclude = ValueItems.merge(exclude, _exclude)
        else:
            exclude = exclude | _exclude if exclude else _exclude

        return super()._iter(to_dict=to_dict, by_alias=by_alias, include=include,
                             exclude=exclude, exclude_none=exclude_none,
                             exclude_unset=exclude_unset, exclude_defaults=exclude_defaults)

    def __init__(self, name: str | RT = None, **cfg):
        """
        Create instance of the resource from another instance
        OR its `name` and other configuration parameters.

        There are 3 modes of the initialization:
          1. from another instance provided as `name` argument (copy constructor)
          2. from the provided configuration arguments ``(name="Name", **cfg)``
          3. from the configuration arguments found by the ``(name)`` from the registry.

        - If resource of *this name is already registered*, AND `cfg` if provided, ``ResNameError`` is raised.
        - If `cfg` is empty and the `name` is not registered, ``ResNotFoundErr`` is raised, unless
          constructor is called with ``_allow_empty=True``.

        That reflects the fact that normally resources are needed to carry information
        in their corresponding attributes beyond its own `name` (which is not truly a resource attribute,
        mainly reserved for the management for references).

        :param name:  name of the created or queried resource OR complete model (copy constructor)
        :param cfg:   the rest of its parameters (if new is created!)
        """
        # first try to identify existing resource
        # object.__setattr__(self, self._file_info_attr, None)
        res = None  # starting from no source resource availabe

        if isinstance(name, self.__class__):  # Resource(res_instance)
            res = name  # act as copy constructor
            if cfg:
                raise RuntimeError("Resource constructor received both model instance and config!")
        elif not cfg and not (res := self._manager.find_resource(name, valid=True)):
            _log.debug(f"{self._manager} failed to locate {self._kind_name} to construct just by {name = }")

        if res:  # copy-construction
            self._deep_copy_attrs(res)  # do not validate again
        else:  # config is provided or not name in registry - a new resource should be created
            if cfg and self._manager.is_listed(name):  # already registered under name
                _log.warning(f"Creating resource {self._kind_name} with already registered {name=}!")
            super().__init__(name=name, **cfg)

        assert hasattr(self, self._file_info_attr)

    # def _str_form(self, exclude=None, *args, **kwargs):
    #     exclude = set(as_iter(exclude)) | {self._file_info_attr}
    #     return super()._str_form(*args, **kwargs, exclude=exclude)

    @classmethod
    def manager(cls) -> ResManager:
        return cls._manager

    @classmethod
    def from_config(cls, cfg: str | dict | RT | Path, *, ignore=False, undefined=True) -> RT:
        """
        Create a new instance of resource from config provided in form of:
          1. name of already registered configuration (creates a copy)
          2. existing resource instance (creates a copy)
          3. dict with actual configuration of the resource.

        >>> ResourceModel.from_config('name')        # find in registry
        >>> ResourceModel.from_config(res)           # res: Model - construct a deep copy - no validation
        >>> ResourceModel.from_config(dct)           # validate dict args
        >>> ResourceModel.from_config(dct, ignore=True, undefined=False) # drop unknown and undefined keys

        :param cfg: name of registered resource, or dict config, or instance of model
        :param ignore: if ``True`` - ignore unknown fields in the given dict
        :param undefined: if ``False`` - drop undefined fields (=None) from arguments
        :return:
        """

        if isinstance(cfg, str) and ('\n' in cfg and ': ' in cfg):
            cfg = cls.parse_raw(cfg)
        elif represents_path(cfg):
            cfg = cls.parse_file_to_dict(cfg)  # continue below with cfg as dict

        if isinstance(cfg, (str, cls)):  # cfg must be a name here!
            return cls(cfg)

        if isinstance(cfg, dict):
            known = cls.__fields__
            if ignore:
                cfg = {k: v for k, v in cfg.items() if k in known}
            if not undefined:
                cfg = drop_undef(ns=cfg)
            return cls(**cfg)

        raise TypeError(f"Unexpected config type={type(cfg)} for {cls}")

    @classmethod
    def from_resource(cls, base: str | RT, **kwargs):
        """
        Constructs new resource with altered attributes from base resource.
        :param base: existing resource in form of string or ResourceModel
        :param kwargs: altered attributes in form of keyword arguments
        :return: new ResourceModel with altered attributes
        """
        if not isinstance(base, cls):
            base = cls.find_resource(base, fail=True)
        kws = base.dict() | kwargs  # FixMe passing base as model ( no kwargs), creates different model.
        return cls(**kws)

    @classmethod
    def is_locatable(cls):
        return bool(ModelsManager.get(cls))

    @classmethod
    def parse_file_to_dict(cls, path: Union[str, Path], *,
                           content_type: str = None, encoding: str = "utf-8",
                           proto: 'ExtendedProto' = None, allow_pickle: bool = False):
        obj = super().parse_file_to_dict(path) or {}
        obj[cls._file_info_attr] = _FileInfo(path)
        return obj

    @classmethod
    def find_resource(cls: RT, name: str, *,
                      as_cfg=False, auto_scan: AutoScan = 'first', fail=False,
                      valid: bool | dict[Type[ResourceModel], list[str]] = True
                      ) -> RT | None:
        """
        If resource registered under the given `name` is found - return it, otherwise `None`.

        :param name: registered name str
        :param as_cfg: return in form of config, not instance
        :param valid: set False to avoid possible validation and consequent rise of error.
        :param fail: if True raise `ResNotFoundError`.
        :param auto_scan: [always, never, first, empty] - conditions to allow to folders rescan if not found.
        """
        return cls._manager.find_resource(name, as_cfg=as_cfg, valid=valid, fail=fail, auto_scan=auto_scan)

    @property
    def is_listed(self) -> bool:
        """Check if this resource instance is registered"""
        return self._manager.is_listed(self.name)


RT = TypeVar("RT", bound=ResourceModel)


class AutoScan(CaseInsEnum):
    ALWAYS = 'always'
    NEVER = 'never'
    FIRST = 'first'
    EMPTY = 'empty'


class ResManager(Generic[RT]):
    """
    Manages ResourceModel subclass of specific type.
    Has various config related functionalities such as:

    * Instantiate new instance.
    * Set the folders of the configurations.
    * Scan those folders in order to collect the configurations.
    * List the scanned configurations.
    * Find a configuration by name.
    """

    class RegInfo:
        """Helper structure used to store resources in the registry"""
        name: str
        cfg: dict
        res: RT
        labels: dict[str, Any]
        _model: RT

        @classmethod
        def _fields(cls) -> set[str]:
            """Defined RegInfo fields"""
            return {f for f in cls.__annotations__ if not f.startswith("_")} | {"path"}

        @property
        def path(self):
            return self.labels and self.labels.get('path', None)

        @timed(_log, cond='DEBUG', form='⏰RegInfo took {time:.4f} sec'.format)
        def __init__(self, model: Optional[Type[RT]] = None, *,
                     cfg: Optional[dict] = None,
                     res: RT | bool = True,
                     labels: Optional[dict] = None):
            """
            Initialize resource RegInfo with ONLY ONE of those: `cfg`, `res`, `labels`.

            The most straightforward way is to provide `res` as ``ResourceModel`` instance.
             - That automatically informs about its model type, so that `model` is unnecessary.

            All other initializations (using `cfg` or `labels`) require `model` argument.
             - In those cases `res` ``= True`` requests model validation,
             - otherwise `res` attribute remains ``None``, useful to postpone validation of
               references to undiscovered resources.

            :param model: Resource class (required unless `res` is provided)
            :param cfg: configuration dict
            :param res: resource instance
            :param labels: dict including the "path" item to the configuration file
            """
            self.labels = labels
            is_res = isinstance(res, ResourceModel)

            if not model:  # ensure model class of this resource
                if not is_res:
                    raise ValueError("Either `model` or `res` instance must be provided")
                model = type(res)
            elif is_res and model is not type(res):
                raise ValueError(f"Mismatch between resource and model types: {type(res)} != {model}")
            self._model = model

            if path := self.path:  # convert path -> cfg
                if cfg or is_res:
                    raise ValueError("`path` can't be provided with `cfg` or `res`!")
                cfg = model.parse_file_to_dict(path)

            if cfg:  # try to convert cfg -> res if requested and possible
                if is_res:
                    raise ValueError("Both `cfg` AND `res` was provided!")
                if res is True:  # request to initialize
                    try:
                        res = model.validate(cfg)
                        cfg = res.dict()
                    except ResNotFoundError as ex:
                        _log.info("Failed initialization attempt: %s", ex)
                        res = None
            elif not is_res:
                raise ValueError("Neither of `path`, `cfg`, `res` was provided!")
            else:
                cfg = res.dict()  # Consider: Do we need a copy of config for homogeneity?

            self.cfg = cfg
            self.res = res
            self.name = cfg and cfg.get('name', None) or res and res.name

            if not self.name:
                raise ResNameError("Can't register resource without name!")

        def validate(self):
            """Validate resource if needed and update cfg.
            :return: res
            """
            if not self.res:
                self.res = self._model.validate(self.cfg)
                self.cfg = self.res.dict()
            return self.res

        def __repr__(self):
            state = '🗸' if self.res else '?'

            def short(s):
                """shorten version of the string"""
                return s if len(s) < 40 else f"{s[:10]}…{s[-24:]}"

            labels = (
                "|".join(f"{k}:{short(str(v))}" for k, v in self.labels.items())
                if self.labels else ""
            )
            return f"{state}{self.name} ⟨{labels}⟩"

    def __init__(self, model: Type[RT]):
        """
        :param model: Pydantic Model class of this resource
        """
        self._resreg: dict[str, ResManager.RegInfo] = {}
        self._scans = 0  # rescanning counter
        self._last_scan: dict = {}  # last scan record: signature, time
        self._scan_par: dict | None = None  # set by `locatable`
        self._locs = Locator()
        self._auto_scan = AutoScan.FIRST
        self.model = model
        self._kind_name = model._kind_name

    def __bool__(self):
        return bool(self._resreg)

    def _set_loc(self, folders, *, cache=False, auto_scan: str | AutoScan = 'first', **search_kws):
        """
        ``pattern`` must match full file path without root folder (from folders).

        Can use * to match folders above:  ``*/some/file.yml``.

        :param folders: folders to search for the config files
        :param auto_scan: when rescan folders on ``find_config`` call
        :param cache: scheme's cache argument - usually ``False`` for resource catalogs
        :param search_kws: arguments for ``inu.datacats.scan.GuideScan``

        Arguments of ``inu.datacats.scan.GuideScan``:
        ::
            pattern: regular expression to search or simplified forms (see above)
            ignore_case: pattern match is case-sensitive
            method: math as: full path | root-relative part | any path at the end
            skip_folders: regex describing `names` of folders to skip (default skips starting with . or _)
            skip_after_match: Skip other file in the folder
            skip_under_match: Skip folders under matched files

            max_depth: Skip folders with larger depth under the root
            max_folders:Skip folders containing many sub-folders
            max_files: Skip folders containing many files
        """
        # important scheme parameters defined explicitly to appear in the interface
        # add them into the dict of scheme arguments (kwargs):
        from toolbox.datacast.scan import GuideScan

        if not (pattern := search_kws.pop("pattern", None)):
            if not (pattern := self.model.format_match_pattern()):
                raise ValueError("Locatable resource requires pattern if format not defined in its model")
        elif isinstance(pattern, re.Pattern):
            pattern = pattern.pattern
        else:  # Consider: add created Format to the model?
            pattern = pm.make_yml_model_format('_', patterns=pattern, model=self.model).match_pattern

        if (method := search_kws.pop('method')) == 'end':
            pattern = f"(.*/)?{pattern}"

        self._scan_par = dict(
            name=f"_Discover<{self.model._kind_name}>",
            scheme=dict(search=GuideScan(pattern=pattern, method=method, **search_kws)),
            **drop_undef(cache=cache)
        )
        self.set_folders(*as_iter(folders))
        self.auto_scan = AutoScan(auto_scan)

    # ToDo: not updated when setting new folders. Reproduce and fix
    def folders(self, order: str | None = None) -> list[Path]:
        """Return folders where resources are being discovered by default"""
        return [*self._locs.existing(order=order)]

    @property
    def auto_scan(self):
        return self._auto_scan

    @auto_scan.setter
    def auto_scan(self, value):
        self._auto_scan = AutoScan(value)

    def set_folders(self, *folders: str | Path,
                    append=False, rescan=False,
                    missing: Literal["ignore", "drop", "fail"] = "ignore"):
        """
        Set folders where resources will be searched for.

        :param folders: folders to add
        :param append: append to or replace the current folders
        :param missing: what to do if a folder is missing from the file system:
                'ignore' and still add, 'drop' it from the list, or
                'fail' `NotADirectoryError` if a folder does not exist.
        :param rescan: if True rescan set folders
        :return the final number of folders
        """
        if missing != 'ignore':
            folders = [*Locator(*folders).existing(fail=(missing == 'fail'))]
        if append and self._locs:
            self._locs += folders
        else:
            if len(folders) == 1 and isinstance(folders[0], Locator):
                self._locs = folders[0]
            else:
                self._locs = Locator(*folders)

        if rescan:
            self.discover()

    # ToDo: Add tests for from_folders and remove_folders
    def from_folders(self, *folders: PathT, inverse=False) -> dict[str, Path | None]:
        """
        Return currently registered resources with paths in given folders.

        :param folders: list of folders to search in
        :param inverse: if set True, return NOT in those folders
        :return: dict {name: folder}
        """
        folders = list(map(normalize, folders))

        def in_folder(path):
            for f in folders:
                if Path(path).is_relative_to(f):
                    return f

        def accept_cond(found: PathT | None):
            return found if not inverse else not found

        return {name: p
                for name, item in self._resreg.items()
                if accept_cond(p := in_folder(item.path))}

    def remove_folders(self, *folders, inverse=False, resources=True, locations=True):
        """
        Remove given folders:
         - from internal search folders list (``self.folders``), and or
         - configs in those folders from the registry

        Reset discovery records if resources are removed.

        :param folders: collection of folders to remove
        :param inverse: if True invert the search for those NOT in folders
        :param resources: ``True`` to remove configs under those folders
        :param locations: ``True`` to remove folders from the search locations of this manager
        :return:  found {name: folder}
        """
        found = self.from_folders(*folders, inverse=inverse)
        if resources:  # remove found resources
            self.remove(*found)

        if locations:  # remove
            self._locs -= folders
        return found

    def remove(self, *names, remove_all: bool | None = None):
        """
        Remove given names from the registry. Reset discovery records if removed.
        :param names: names to remove
        :param remove_all: boolean that indicates whether to remove all configurations.
        """
        if remove_all:
            if names:
                raise ValueError("Provided both names and remove_all=True")
            self._resreg = {}
        else:
            for k in names:
                del self._resreg[k]
        if remove_all or names:
            self.last_discover(tm=0)

    def add_resource(self, res: RT | dict | PathT | RegInfo, *, over=False):
        """
        Add resource provided in form of instance, config dict, or path to its config file.
        :param res: resource to add
        :param over: if False raise ``ResNameError`` if the name is already listed.
        """
        if not isinstance(res, self.RegInfo):
            arg = ({'res': res} if isinstance(res, ResourceModel) else
                   {'cfg': res} if isinstance(res, dict) else
                   {'labels': {'path': res}})
            res = self.RegInfo(self.model, **arg)

        if res.name in self._resreg and not over:
            raise ResNameError(f"Resource {self.model.__name__} named {res.name} is already listed!")
        self._resreg[res.name] = res
        _log.debug("⟴ %s %s", self.model.__name__, res)

    def last_discover(self, folders: list[PathT] | None = None, tm: float | None = None,
                      report: bool | Literal['age', 'time', 'full'] = False) -> float | None | str:
        """Sets (if `tm` is provided) or gets time of the last discovery record.

        Keep only record of last discovery including signature of used folders.
        If passed `tm` is **0**, resets discovery history

        :param folders: list of folders used in discovery, by default - internally defined ones
        :param tm: time moment of last discovery of *those folders*
        :param report: return the last discovery report as string or just timestamp if `False`
        :return: None when sets, 0 if given folders were never discovered.
        """
        from toolbox.utils.strings import hash_str

        if tm == 0:
            if folders:
                raise ValueError("Passing folders when requesting reset with tm=0 is forbidden!")
            _log.info(f'Resetting resource {type(self)} discovery record at scans = {self._scans}')
            self._scans = 0
            self._last_scan = {}
            return

        signature = hash_str(''.join(map(str, folders or self.folders())))
        if tm is None:  # GET last discovery!
            last_disc = self._last_scan.get(signature, 0)
            if not report:
                return last_disc
            if last_disc:
                from datetime import datetime
                dtm = datetime.fromtimestamp(last_disc).strftime('%H:%M:%S ') \
                    if report in {'time', 'full', True} else ''
                elapsed = f'({time.time() - last_disc:.1f}sec ago)' \
                    if report in {'age', 'full', True} else ''
                return dtm + elapsed
            return f'Undiscovered!'

        self._scans += 1
        self._last_scan = {signature: tm}
        return

    # ToDo: check why discover doesn't work with '-' separated config names
    @timed(_log.info, cond=__debug__, min=0.1,
           pre="🏳{func_name}({args[0]},...) started ...".format,
           form='⏰{func_name}({args[0]}, ...) call took {time:.1f} sec'.format)
    def discover(self, *folders: PathT, loc_order: str | None = None,
                 append=False, only=False, fail=False,
                 cache: Optional[bool] = None, progress=False):
        """
        Rescans folders associated with this config manager, and/or
        Can fail on unsuccessful scan - which means:
            1. no configurations are found on the instance folders.
            2. corrupt configuration is found and terminated discovery

        :param folders: optional additional folders to use in this scan
        :param append: replace or append newly found resources
        :param loc_order: locations order from `Locator` codes 'IEA' (Internal, Environ, Additional)
        :param only: True to scan only in the specified `folders`, False for search also in `self.folders`
        :param fail: if False `log.error` instead of raise on resource parsing error
        :param cache: True to enable, False to disable, None to let scheme default decide.
        :param progress: if True show collection progress
        """
        from toolbox.datacast.collect import DataCollection
        from toolbox.datacast.caster import DataCaster

        model_name = self.model.__name__
        mode = append and 'append' or 'refresh'

        folders = [normalize(f) for f in as_list(folders)]
        if not only:
            folders += self.folders(order=loc_order)
        if not folders:
            _log.error(f'Discovery failed for {model_name} - no existing folders found by\n\t {self._locs}')
            return 0

        prev_sz = len(self)
        _log.debug("Creating `%s` discovery datasets for folders %s...", self._kind_name, folders)
        resource_scanners = [DataCaster(**self._scan_par, source=p) for p in folders]
        dc = DataCollection("DiscoveredResources", datasets=resource_scanners,
                            **drop_undef(cache=cache), progress=progress)
        records = dc.db.reset_index()[[*dc.categories]].to_dict(orient='records')

        if not append:
            self.remove(remove_all=True)

        _log.debug(f'... found {len(records)} `%s` configs to add ...', self._kind_name)
        for rec in records:
            try:
                self.add_resource(self.RegInfo(self.model, labels=rec))
            except (ResError, ValidationError) as ex:
                if isinstance(ex, ValidationError):
                    ex = ex.errors()[0]['msg']
                _log.error("%s adding %s(%s):\n\t%s", type(ex).__name__, model_name, rec, ex)
                if fail: raise ex
        self.last_discover(folders, time.time())

        (_log.info if (new_sz := len(self)) and new_sz >= prev_sz else _log.warning)(
            'Discovered (%s) %d (was %d) configs of <%s> in %s',
            mode, new_sz, prev_sz, model_name, folders
        )
        return len(self._resreg)

    def list(self, attr: Literal["name", "path", "cfg", "res", "labels"] | None = None,
             columns: bool | str | Collection[str] = False):
        """
        List information of registered resources in form of either
        ::
            ResManager.RegInfo:
                name: str
                path: str
                cfg: dict
                res: ResourceModel
                labels: dict[str, Any]
        or one of its attributes if `attr` is specified

        Return a ``list`` of requested ``RegInfo`` *attributes* or ``DataTable``:

        - `(None, False)`: ``list`` of the `RegInfo` objects
        - `(attr, False)`: ``list`` of the specffied attribute of the `RefInfo` object
        - `(attr, True)`: ``DataTable`` of the specified attribute of the `RefInfo` object as column
        - `(None | cfg, True)`: ``DataTable`` with model's fields as columns
        - `(None | cfg, [field, ...])`: ``DataTable`` with specified model's fields as columns

        Those fields are declared on the YamlModel subclass definition.

        :param attr: which attribute of `RegInfo` objects to list, None - list `RegInfo` objects
        :param columns: if True or model filed(s), return `DataTable` instead of the list.
        """
        if not (attr is None or attr in (allowed_attributes := self.RegInfo._fields())):
            raise ValueError(f"Argument {attr=} must be `None` of in {allowed_attributes}!")
        if columns and not attr:
            attr = "cfg"
        if isinstance(columns, (str, tuple, list)):
            columns = as_list(columns)
        elif not isinstance(columns, bool):
            raise ValueError(f"Invalid argument {columns=}")

        add_file_attrs = False  # should we add file attributes as columns
        def_file_attrs = {k: None for k in _FileInfo.__fields__}  # defaults when no file source
        if isinstance(columns, list):
            if attr != "cfg":
                raise ValueError('Columns names may be provided only if attr is "cfg" or None')
            add_file_attrs = bool(set(columns).intersection(def_file_attrs))

        def extract_attr(info):
            if not attr:
                return info
            attrs = getattr(info, attr)
            if add_file_attrs:
                attrs |= attrs.pop(self.model._file_info_attr, None) or def_file_attrs
            return attrs

        out = [extract_attr(info) for info in self._resreg.values()]
        if not columns:
            return out

        from toolbox.utils.pdtools import DataTable
        df = DataTable(out, columns=[attr] if attr in ("name", "path", "res") else None)
        if isinstance(columns, list):
            df = df[columns]
        return df

    _OUT_CFG_T = Union[str, RT, dict]

    def _need_rescan(self, auto_scan: Optional[AutoScan] = None) -> bool:
        """Return ``True`` if ``auto_scan`` conditions require rescan"""
        auto_scan = AutoScan(auto_scan) if auto_scan else self.auto_scan
        return (auto_scan is AutoScan.ALWAYS or
                auto_scan is AutoScan.FIRST and self._scans == 0 or
                auto_scan is AutoScan.EMPTY and len(self) == 0)

    def is_listed(self, name) -> bool:
        # Fastest way to check if the name is in the registry
        return name in self._resreg

    def find_resource(self, name: str, *, valid=True, fail=False,
                      as_cfg=False, auto_scan: AutoScan = None,
                      fuzzy: bool | int | None = None,
                      ) -> None | _OUT_CFG_T | list[_OUT_CFG_T]:
        """Find registered resource by name and return as instance or its VALIDATED config.

        `fuzzy` argument allows for imperfect case-insensitive queries:
            - `False`: disable
            - `True`: enable to accept name similar > 90%
            - `int`: from 0-100 (%) score threshold to accept
            - `None`: if perfect match not found only reports possible close matches

        :param name: unique name in the catalog or None to return ALL the configurations
        :param as_cfg: return as config or as resource instance
        :param fuzzy: `True` - find best match
        :param valid: ensure validation of found resource - raise Exception if fails!
        :param fail: if not found raise `NameError` instead of returning ``None`` or ``{}``
        :param auto_scan: override optionally instance auto-scan policy
        """
        if not name:
            msg = f"Requested resource without {name=}"
            if fail:
                raise NameError(msg)
            _log.debug(msg)
            return

        if name not in self._resreg:  # possible rescan if not found
            need = self._need_rescan(auto_scan)
            _log.debug(f"Resource {self._kind_name}[{name}] not in registry, discovery {need=}")
            need and ModelsManager.discover(self.model, dependencies=True, append=True)

        info = self._resreg.get(name, None)  # another attempt after rescan
        if not info and fuzzy is not False and self._resreg:
            from toolbox.utils.strings import fuzzy_find
            names = list(self._resreg)
            if fuzzy is None:  # just find closest and report of possible candidate
                if found := fuzzy_find(name, names, out='string', case=False, score_cutoff=92):
                    _log.warning(f'{self} contains "{found[0]}" similar to the requested "{name}"')
            else:
                if fuzzy is True:
                    fuzzy = 90
                if found := fuzzy_find(name, names, out='index', case=False, score_cutoff=fuzzy):
                    info = self._resreg[found[0]]
        if info:
            if valid or not as_cfg:
                info.validate()
            return info.cfg if as_cfg else info.res

        if fail:
            available = ', '.join(self.list('name'))
            raise ResNotFoundError(f"No {self.model._kind_name} with {name=} among {available = }")

    def __str__(self):
        return f"<RM:{self.model.__name__.replace('Model', '')}>[{len(self)}]"

    def __repr__(self):
        from toolbox.utils.strings import short_form
        locs = short_form(str(self._locs.first), 10, 16)
        return f"{self} in {locs}, {self.last_discover(report=True)}"

    def __contains__(self, name):
        return name in self._resreg

    def __len__(self):
        return len(self._resreg)

    def __getitem__(self, item: int | str):
        """Return resource from the registry by its nume or number"""
        if isinstance(item, int):
            item = [*self._resreg][item]
        return self._resreg[item].validate()


def dependency_order(models: Collection[ResourceModel], *, extend=False, fail=False
                     ) -> list[ResourceModel] | None:
    """Produce topological order of the given resource models.

    :param models: collection of resource models to order
    :param extend: if ``True`` add to the result all the dependencies
    :param fail: if ``True`` raise ``graphlib.CycleError`` if cycles are detected
    """
    from graphlib import TopologicalSorter, CycleError

    # traverse all the tree of references
    to_check = set(models)  # nodes to search for references
    checked = set()  # track all the nodes once checked to ensure that
    while to_check:  # any final tree is eventually exhausted (nothing to check)
        refs = set(sum((m._refers for m in to_check), []))  # found references
        checked.update(to_check)  # move from to_check into checked
        to_check = refs - checked  # select for checking only new from the found references

    graph = {m: {r for r in m._refers} for m in checked}
    ts = TopologicalSorter(graph)
    try:
        order = list(ts.static_order())
        if not extend and len(models) < len(order):
            keep = set(models)
            order = [m for m in order if m in keep]
        return order
    except CycleError as ex:
        _log.error(f"Resources in cycled dependency: {ex}")
        if fail:
            raise ex
    return None


class ModelsManager:
    """
    Manages the ManagedModels that are resources.
    The instance of specific ResourceModel type is a configration.
    Each ResourceModel type has singular ConfigManager that manages its configurations.
    The ModelsManager acts as a ConfigManager Factory.
    It holds a registry of each model and the ConfigManager that manages it.

    The constructor interface accepts:
    * The ResourceModel type.
    * The scheme that will be applied during folder scanning.
    * The folders where the configurations live.
    * Additional keyword arguments will be passed as inline labels to the scheme constructor.

    ModelsManager allows the user various actions on the registered elements:
    * Finding a ConfigManager of specific ResourceModel type.
    * Finding a scanned configuration of ConfigManager of specific ResourceModel type.
    * Rescanning all the folders of the registered ConfigManagers.
    * Listing all the registered ConfigManagers
    """

    _res_managers: dict[Type[RT], ResManager[RT]] = {}
    _res_names: dict[str, Type[RT]] = {}

    @classmethod
    def set_env_loc(cls, env: str | Path | bool = None):
        """Sets global environment locators.
         Convenient for some case, over generally preferable `inu.env.setup()`.

          - ``None``: sets to inu.env defaults ONLY if is currently unset
          - ``True``: sets to the inu.env defaults
          - `str` | `Path` - name or path of .env file with settings
        """
        from inu.env import EnvLoc
        if env is False or env is None and EnvLoc.last_set():
            return
        EnvLoc.reset(None if env is True else env)
        return EnvLoc

    @classmethod
    def discover(cls, *models: str | RT, dependencies=True, append=None,
                 cache: bool = None, aging=2.):
        """
        Discover resources of specified (if provided otherwise all the registered) resource models.

        Some resources may reference and therefore depend on other types of resources,
        which preferable should be discovered before them.

        To take into account those dependencies and
        discover resources in the proper order automatically use ``dependencies=True``.

        Otherwise, discover only listed models in the given order.

        :param models: names or classes of the Resource Models to discover resources for
        :param dependencies: order discovery of all the interdependent resources
        :param cache: allow|disable discovery caching or leave as is (None)
        :param aging: min seconds before running discovery of a resource again
        :param set_env: first set environment: name or ``True`` for default.
        """
        cls.set_env_loc()

        managers = models and [cls.get(m) for m in models] or cls._res_managers.values()
        _log.debug('🔭Starting discovery of %s', [m._kind_name for m in managers])

        # Consider: move building dependencies tree into manager class initialization
        if dependencies:  # otherwise just discover all the listed models
            models = {mng.model for mng in managers}  # originally requested models
            order = dependency_order(models, extend=True)

            if dependees := [m._kind_name for m in set(order) - models]:
                _log.info(f'Additionally discovering {dependees=}')

            managers = [m._manager for m in order]  # reordered and extended by dependees

            if aging:
                now = time.time()
                aged = lambda mng: (
                                       age := (now - mng.last_discover())
                                   ) > aging or _log.warning(
                    f"Skipping discovery of {mng._kind_name}: {age=:.1f} < {aging=:.1f} sec"
                )
                managers = list(filter(aged, managers))
        n = 0
        for mng in managers:
            n += mng.discover(**drop_undef(cache=cache, append=append))
        return n

    @classmethod
    def remove_all(cls):
        """Remove all resources discovered by all the managers."""
        for rm in cls._res_managers.values():
            _log.info(f'Removing all the resources from {rm}')
            rm.remove(remove_all=True)

    @classmethod
    def list(cls, *models):
        """Lists all or requested ConfigManagers on the registry."""
        if not models:
            return list(cls._res_managers.values())
        if len(models) == 1:
            return cls.get(models[0]).list()
        return list(map(cls.get, models))

    @classmethod
    def register(cls, model: Type[RT], *, _restore=True) -> ResManager[RT]:
        """
        Register new ``ResourceModel`` by associating with it a new ``ResManager``.

        Raise ``KeyError`` if model class already exists.

        :param model: subclass of ``ResourceModel``
        :param _restore: internal flag to debug special cases
        :return: created ``ResManager`` instance.
        """
        name = f"{model.__module__}.{model.__name__}"
        if model in cls._res_managers or name in cls._res_names:
            _log.error(f"Resource manager for ({name}) is already registered - replacing!")
            old_model = cls._res_names.pop(name, None) or model  # None if found model!
            cls._res_managers.pop(old_model, None)

        if (rm := getattr(model, '_manager', None)) is None:
            rm = ResManager(model)
            _log.info(f'Registered resource model: {name}')
        elif _restore:
            _log.warning(f"Restoring previous {name}'s manager - abnormal situation")
        else:
            raise RuntimeError()

        cls._res_managers[model] = rm
        cls._res_names[name] = model
        dependency_order(cls._res_managers, fail=False)  # check and report on cycles
        return rm

    @classmethod
    def register_models(cls, *sources: RT | ModuleType, over=True, _restore=True):
        """Usually not used! - models are registered automatically as they are imported.

        This may be required for development purposes, for example after reset erased the registry.

        Registers specifically provided ``ResourceModels`` or found in provided modules.

        :param sources:
        :param over:
        :param _restore:
        """
        if _log.isEnabledFor(_log.INFO):
            sources_str = "\n\t".join(
                src.__name__ if isinstance(src, ModuleType) else str(src)
                for src in sources
            )
            if len(sources) > 1:
                sources_str = f"sources: \n\t{sources_str}"
            _log.info("Registering resource models from %s", sources_str)

        for src in sources:
            try:
                if isinstance(src, ModuleType):
                    for m in src.__dict__.values():
                        if isinstance(m, type) and issubclass(m, ResourceModel) and m is not ResourceModel:
                            cls.register(m, _restore=_restore)
                elif isinstance(src, ResourceModel):
                    cls.register(src, _restore=_restore)
                else:
                    raise ValueError(f"{src} is neither ResourceModel nor python module")
            except ResModelError as ex:
                _log.error(ex)
                if over:
                    raise ex

    _MatchModel = enum_attr('__call__',
                            start=str.startswith,
                            part=str.__contains__,
                            full=str.__eq__,
                            name=lambda x, y: x.replace('model', '') == y)

    @classmethod
    def get(cls, model: Type[RT] | str, *,
            match: str | _MatchModel = _MatchModel.part) -> ResManager[RT] | None:
        """Return resource manager associated with given model.

        Model is found either by its class or name (case-insensitive).

        Class name match strategies:
         - part - any part of the name is matched
         - start - only match the start of the name
         - name - match after removing possible 'Model' part of the class name
         - full - full class name match

        :param model: class of the model or ist name, or some part of it
        :param match: kind of match expected if model is a string.
        """
        if isinstance(model, str):
            match = cls._MatchModel(match)
            if not (name := model.lower()):
                raise ValueError('Empty Model name requested')

            for kls, v in cls._res_managers.items():
                if match(kls.__name__.lower(), name):
                    return v
        elif isinstance(model, int):
            sz = len(cls._res_managers)
            model = list(cls._res_managers)[model] if -sz <= model < sz else None
        return cls._res_managers.get(model, None)

    @classmethod
    def _reset(cls):
        """For development purposes: resets the resource managers registry"""
        _log.warning("Resetting Resource Models Manager - Not a normal request!")
        cls._res_managers = {}
        cls._res_names = {}

    @classmethod
    def find(cls, model: RT | str, name: str | None = None,
             as_cfg: bool = False, auto_scan: AutoScan = None, fail=False
             ) -> ResManager | str | ResourceModel | DataFrame | DataCollection | dict | None:
        """
        Finds desired configuration using the appropriate ResourceManager.

        Acts as a shortcut to
        ::
            find_config_manager(model).find_config(name, out=...)

        or, if `name` is not provided, then return the ``ConfigManager``:
        ::
            find_config_manager(model)

        :param model: the model class
        :param name: the configuration name
        :param as_cfg: return as config or as resource instance
        :param fail: if True rise KeyError instead of returning ``None``
        :param auto_scan: override instance policy: 'always', 'never', 'first', 'empty'
        :return: the configuration if found else None
        """
        cm = cls.get(model)
        if name is None:
            return cm
        return cm.find_resource(name, as_cfg=as_cfg, auto_scan=auto_scan, fail=fail)

    def __repr__(self):
        from inu.env import EnvLoc
        env_info = EnvLoc.repr(True)
        managers = '\n'.join(map(repr, self.list()))
        max_len = lambda s: len(max(s.split('\n'), key=len))

        sep = '━' * max(max_len(env_info), max_len(managers))
        return f"{env_info}\n{sep}\n{managers}\n"

    def __getitem__(self, item) -> ResManager:
        return self.get(item)


def find_resource(kind: str | ResourceModel, name: str = None, auto_scan=AutoScan.NEVER, fail=False):
    """
    Find resource of specific kind with specific name.

    If name not provided return the ConfigManager of given ``kind``.

    :param kind: Model class of resource, or part of its name
    :param name: registered name of the resource
    :param auto_scan: condition to initiate folders scan: 'always', 'never', 'first', 'empty'
    :param fail: Should it fail
    :return:
    """
    return ModelsManager.find(kind, name, auto_scan=auto_scan, fail=fail)


def locatable(
        folders: PathT | Locator | list[PathT] | None = None,
        pattern: Optional[str | re.Pattern] = None,
        ignore_case: bool = False,
        method: Literal["relative", "full", "end"] = "end",
        skip_folders: str = r"^[._].*",
        max_depth: int = 4,
        skip_after_match: bool = False,
        skip_under_match: bool = False,
        max_folders: Optional[int] = None,
        max_files: Optional[int] = None,
        cache: bool = False,
) -> Callable[[Generic[RT]], Type[RT]]:
    r"""
    Decorator for a subclasses of ``ResourceModel`` to configure discoverability
    of its configration (*resource*) files.

    Resource files are searched folders defined as list of paths or ``Locator`` objects,
    and according to (optionally provided) ``GuideScan`` arguments, grouped into:
       - Matching parameters: `pattern`, `method`, `ignore_case`
       - Scanning strategies: `max_depth`, `skip_folders`, `skip_after_match`, `skip_under_match`, ...

    **Search Pattern**

    The `pattern` argument defines a regular expression matching, depending on the
    ``method`` arguments, either:
      - the `full` path
      - its part `relative` to a *root folder*
      - only the `end` of the path

    Pattern be provided as:
      - explicitly compiled regex ``re.Pattern``:  `re.compile("some_\\d{2}\\.dat")
      - string automatically converted into regex: `".some.yml"` -> "\\.some\\.yaml"
      - regex in str form `"\\.some\\.yaml$"` (terminating `$` helps to differentiate from a string)

    **Example:**

    Finds all the files with extension `.some.yml` in the tree up to ``max_depth`` deep
    under `/somewhere/proj/res_folder`:
    ::
        @locatable(folders='/somewhere/proj/res_folder', pattern='.some.yml')
        class SomeModel(ResourceModel):
             ...

    :param folders: root folders where resource files should be searched for

    :param pattern: regular expression to search
    :param ignore_case: pattern match is case-sensitive
    :param method: math as: full path | root-relative part | any path at the end

    :param skip_folders: regex describing `names` of folders to skip, by default starting with ``._``
    :param skip_after_match: Skip other file in the folder
    :param skip_under_match: Skip folders under matched files
    :param max_depth: Skip folders with larger depth under the root
    :param max_folders:Skip folders containing many sub-folders
    :param max_files: Skip folders containing many files

    :param cache: search cache (``DataCaster``) argument - normally ``False`` for resource catalogs
    """
    loc_par = locals().copy()
    from toolbox.datacast.scan import GuideScan
    from toolbox.utils.datatools import rm_keys

    if folders is None:  # Consider: rase Exception?
        _log.error(f"Resource is specified `locatable` without specifying locations!")

    loc_par = rm_keys(loc_par, ['folders', 'cache'])
    assert not (_ := set(loc_par).difference(GuideScan.__fields__)), f"Unknown GuideScan args {_}"
    loc_par = drop_undef(**loc_par)

    def model_decor(model: Type[RT]):
        model._manager._set_loc(folders, cache=cache, **loc_par)
        return model

    return model_decor


resman = ModelsManager()
