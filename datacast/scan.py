from __future__ import annotations

import os
from time import time
from typing import Literal, Optional

import pydantic
import regex as re
from pydantic import PrivateAttr

from toolbox.param import YamlModel
from toolbox.utils import logger
from toolbox.utils.filesproc import normalize
from toolbox.utils.paths import TransPath

_log = logger('datacast.scan')


class GuideScan(YamlModel, hash_exclude=['_pather', '_match_skip_folders']):
    """
    Constrain-guided scanner for folders tree.

    Produces iterator over labels extracted from files matching specific pattern.
    Scanning may is guided by additional search parameters:
    ::
        pattern: regular expression to search
        ignore_case: set case-sensitivity of the matching
        method: math as: full path | root-relative part | any path at the end

        skip_folders: regex describing `names` of folders to skip (default skips starting with . or _)
        skip_after_match: Skip other file in the folder
        skip_under_match: Skip folders under matched files

        max_depth: Skip folders with larger depth under the root
        max_folders:Skip folders containing many sub-folders
        max_files: Skip folders containing many files

    """
    Field = pydantic.Field

    # -------------------------------------
    pattern: str
    test_url: Optional[pydantic.AnyUrl]
    ignore_case: bool = False

    method: Literal['relative', 'full', 'end'] = Field('relative', description=(
        'Match pattern with: full path | root-relative part | any path at the end'))
    skip_folders: str = r"^[._].*"
    skip_after_match: bool = Field(False, description="Skip other file in the folder")
    skip_under_match: bool = Field(False, description="Skip folders under matched files")

    max_depth: int = Field(4, description="Skip folders with larger depth under the root")
    max_folders: int = Field(None, description="Skip folders containing many sub-folders")
    max_files: int = Field(None, description="Skip folders containing many files")

    samples: list[str] = Field(None, description="List of sample paths for validation")
    # ---------------------------------------------------
    _pather: PrivateAttr = PrivateAttr()
    _match_skip_folders: PrivateAttr = PrivateAttr()

    def verify_samples(self, fail=False):
        """Return list of sample failed parsing"""
        total = self.samples and len(self.samples) or 0
        failed = [s for s in self.samples if self._pather.regex.parse(s) is None] if total else []
        if num := len(failed):
            msg = (f"Pattern {self._pather.regex.regex.pattern}\n\tfailed parsing "
                   f"{num} of {total=} samples:\n\t{failed}")
            if fail:
                raise ValueError(msg)
            else:
                _log.error(msg)
        return len(failed), total, failed

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        flags = self.ignore_case and re.IGNORECASE or 0
        self._pather = TransPath(self.pattern, flags=flags)
        self.verify_samples(fail=True)
        self._match_skip_folders = self.skip_folders and re.compile(self.skip_folders).fullmatch

    def scanner(self, root, win_to_posix: bool):
        """
        Return generator of labels parsed from matching files under the root
        :param root: path to search under
        :param win_to_posix: if `True` convert files paths into posix before matching and parsing.
                             Allowed only in 'relative' matching method.
        :return: Generator of parsed labels for found matching files.
        """
        root = normalize(root, out=str)
        if win_to_posix and self.method not in (sup := ['relative', 'end']):
            raise ValueError(f"win_to_posix is compatible only if method in {sup}")
        crop = len(root) + 1 if self.method == 'relative' else 0
        method = 'search' if self.method == 'end' else 'fullmatch'
        parse = self._pather.regex.parse

        def match(p: str):
            """Match closure tailored for specific root cropping"""
            if win_to_posix:
                p = p.replace('\\', '/')
            if (m := parse(crop and p[crop:] or p, method=method)) is not None:
                return m | {'path': p}
            return None

        _log.debug(f"🔍Scanning for {self._pather.form} in {root}")
        return self._walk(root, match=match)

    def _walk(self, root, *, match, level=0):
        """
        Iterate over files and/or dirs under the given `root` folder.
        :param root:  the folder to start from
        :param level: level of the caller
        :returns: Iterator over dictionaries with labels extracted by parsing
        """
        # to optimize the "for entries" loop as much as possible all the
        # per entry access calculations are assigned to variables:
        t0 = time()
        ms = lambda: f"{(time() - t0) * 1000:.2f}ms"

        do_log = _log.isEnabledFor(_log.DEBUG)
        max_dirs, max_files = self.max_folders, self.max_files
        skip_after_match, skip_under_match = self.skip_after_match, self.skip_under_match

        any_skip = skip_under_match or skip_after_match
        check_skip_folders = bool(self.skip_folders)
        check_dirs = (level := level + 1) < self.max_depth  # False stops entering sub-folders

        files, folders = [], []
        collect_files = bool(max_files)
        check_files = True
        n_dirs, n_files, n_matches = 0, 0, 0

        def skip():
            nonlocal check_files, check_dirs
            if skip_after_match: check_files = False
            if skip_under_match: check_dirs = False
            if not (check_dirs or check_files):
                do_log and _log.debug(f'Dropped 💧 (after: {skip_after_match}, '
                                      f'under {skip_after_match}) match in 🖿{root}')
                return True
            return False

        with os.scandir(root) as entries:
            n = 0
            for n, entry in enumerate(entries, 1):
                if entry.is_dir():
                    n_dirs += 1
                    if max_dirs and n_dirs >= max_dirs:
                        do_log and _log.debug(f'Dropped 💧 at {n_dirs=} ({ms()}) 🖿{root}')
                        return
                    # check_dirs may be False after skip_under_match!
                    if check_dirs and not (check_skip_folders and self._match_skip_folders(entry.name)):
                        folders.append(entry.path)
                elif check_files:  # block on after match in no-collect mode
                    n_files += 1
                    if max_files and n_files >= max_files:
                        do_log and _log.debug(f'Dropped 💧 at {n_files=} ({ms()}) 🖿{root}')
                        return

                    if collect_files:
                        files.append(entry.path)
                    elif m := match(entry.path):
                        yield m  # process files on the fly
                        n_matches += 1
                        if any_skip and skip(): break

            for path in files:  # if collect_files is False, then files == []!
                if m := match(path):
                    yield m
                    n_matches += 1
                    if any_skip and skip(): break

            do_log and _log.debug(f'{level}⇩ Scanned{n:3d} '
                                  f'({n_dirs}🖿+{n_files}🗏 !{n_matches}) in {ms()} in 🖿{root}')
            if check_dirs and folders:
                t0 = time()
                for folder in folders:
                    yield from self._walk(folder, match=match, level=level)
                do_log and _log.debug(f'⮱ {len(folders)} sub-folders scanned in {ms()} in 🖿{root}')
