from __future__ import annotations

import ast
import csv
from typing import Sequence

import numpy as np
import pandas as pd

from inu.env import EnvLoc
from toolbox.utils.pdtools import DataTable
from pathlib import Path

__all__ = ['IssueCollection']

# ToDo Update to ROICollection
class IssueCollection:
    """
    Container for issues and problems we keep of our algorithms.
    Handle the issues as a datatable with labels and a polygon vertices depicting the ROI in the image.

    Includes a show function to analyze the issues on the images.
    Use the save function to save an updated table to a file.
    """

    # attributes which could be inherited when modifying a collection
    _inherit_attrs = ['name', 'lbl_types']

    @classmethod
    def from_db(cls, db, *, like: IssueCollection = None, cat_labels=None, name=''):
        """
        Create a IssueCollection object from given db and optionally other
        attributes.

        :param db:  DataTable(Frame) object to build the collection from
        :param like: IssueCollection object to inherit some of its attributes
        :param cat_labels: override that in like. Required if like is None!
        :param name: override that in like. Required if like is None!
        :return: a new IssueCollection
        """
        if like is None and not isinstance(name, str):
            raise ValueError("Provide either name or like")

        ic = cls.__new__(cls)
        ic.db = db if isinstance(db, DataTable) else DataTable(db)

        lc = locals()
        for attr in cls._inherit_attrs:
            if not (val := lc.setdefault(attr, None)) and like:
                val = getattr(like, attr)
            setattr(ic, attr, val)
        return ic

    @staticmethod
    def from_csv(file: str, file_mapping: bool = False):
        """
        Loading issue collection from a csv file.
        First row of the file is the types of each column, second row is the name of the label, and than
         each row is a data item.

        Example:
        str,str,issue_type,str,float,tuple
        dataset,    scene,  issue_type,     alg,    id,     polygon
        FT3D,   A_0000_0007,    S2,     alg_1,      10,     "((688, 10), (722, 10), (722, 73), (688, 73))"
        FT3D,   A_0000_0007,    S4,     alg_2,      11,     "((465, 262), (465, 288), (481, 288), (481, 262))"


        Currently categorical labels supported only in Excel (xlsx) files as tables.
         The values for each category label are ALWAYS in column named "Name".

        :param file:
        :return:
        """
        with open(Path(file).expanduser(), newline='') as csvfile: # Read first row to get types
            reader = csv.reader(csvfile)
            types = next(reader) # list

        issues = pd.read_csv(file, skiprows=1)
        lbl_types = {}
        # extract categorical labels from the csv
        for col_num, typ in enumerate(types):
            col_name = issues.iloc[:, col_num].name
            if typ not in [ 'int','float', 'str', 'tuple']: # categorical label
                cat_file = EnvLoc.ISSUES.first_existing() / f'categorical/{typ}.xlsx'
                lbl_types[col_name] = {typ : pd.read_excel(cat_file).set_index('Name')}
            else:
                if typ == 'tuple':  # tuple in csv are saved as string
                    issues[col_name] = issues[col_name].apply(func=ast.literal_eval)
                lbl_types[col_name] = typ
        return IssueCollection(issues, lbl_types, file_mapping=file if file_mapping else None)

    def __init__(self, issues: DataTable | Sequence[dict],
                 label_types=None,
                 file_mapping: str = None,
                 name=None):
        """
        Low level function. For loading a collection use IssueCollection.from_csv()

        :param name: Name of the issue collection. Must be included.
        :param issues: path for an issues table in file, or a DataTable
        :param label_types: dict of types for each label: {lbl_name: type}.
                          For a categorical labels - type is a dict with the type and its categories:
                          {lbl_name: {type : [val1, val2, ...]}}
        :param file_mapping: if not False - name of csv file to keep update when changing the collection
                             False - don't update file automatically.

        """

        self.db = DataTable(issues)
        index = self.db.columns.drop('polygon')
        self.db.set_index(list(index), inplace=True)
        self.name = name or ''
        self.lbl_types = label_types
        self.file_mapping = file_mapping

    def qix(self, *args, drop_level: bool = False,
            axis=None, key_err=True, as_ic=True, **kws) -> DataTable | IssueCollection:
        """Fuzzy query of issue collection index and return either filtered
        version of the issue collection (as_ic argument) or
        filtered `db` table.

        :param args: list of values from one of the index levels
                     - will try all until first is found or raise KeyError
        :param drop_level: if True - drop found levels from the result
        :param axis: if specified query only this axis
        :param key_err: if False ignore key errors
        :param as_ic: results are returned as a new IssueCollection
        :param kws:  {level: value} - eliminates exhaustive search in all levels

        :return: Selected issues as DataTable or IssueCollection.
        """
        # a single unnamed argument could be an index or list of indices
        if not kws and len(args) == 1 and isinstance(args[0], (list, tuple, pd.Index)):
            if isinstance(idx := args[0], tuple):
                idx = list(idx) if isinstance(idx[0], tuple) else [idx]
            sub = self.db.loc[idx]
            if len(sub) != (len(idx) if isinstance(idx, list) else 1):
                raise IndexError("Invalid index passed to qix")
        else:  # query based indexing
            sub = self.db.qix(*args, drop_level=drop_level, axis=axis, key_err=key_err, **kws)

        if not as_ic:
            return sub

        # ----  new IssueCollection parameters ----
        before, after = len(self.db), len(sub)
        if before == after:
            return self
        return type(self).from_db(sub, like=self,
                                  name=isinstance(as_ic, str) and as_ic or self.name)

    def add(self, issue: dict | Sequence[dict] | DataTable | IssueCollection):
        """
        Add issue to the collection
        :param issue: dict with labels of the issue, or a datatable / collection with one or more issues
        :return:
        """

        if isinstance(issue, IssueCollection):
            assert issue.lbl_types == self.db.lbl_types, "Both collections must have the same column types"
            self.db = self.db + issue.db
        else:
            if isinstance(issue, dict):
                issue = [issue]
            self.db = self.db + IssueCollection(issue).db
        self.file_mapping and self.save(self.file_mapping)


    def remove(self, issue: dict):  # ToDo support more issue type options
        """
        Remove issue from the collection
        :param issue: dict with labels of the issue
        :return:
        """
        if 'polygon' in issue.keys(): # polygon is the data, not in the multiindex
            issue.pop('polygon')
        self.db = self.db.rmi(**issue)
        self.file_mapping and self.save(self.file_mapping)


    def __str__(self):
        return f"⚠{type(self).__name__}<{self.name}>[{len(self)}]"

    def __repr__(self):
        if self.empty:
            return f"{self}"
        index_cats = ','.join(self.db.index.names)
        data_cats = ','.join(self.db.columns)
        return f"{self}\n" \
               f"🔖 Index[{index_cats}] ⮕ Data[{data_cats}]"

    def save(self, path: str | Path):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(mode=0o777, exist_ok=True)
        ext = path.suffix.lower()
        if ext == '.csv':
            self._to_csv(path)
        else:
            raise ValueError("Only csv files are supported for saving")

    def _to_csv(self, file: str | Path):
        """
        Saving issue collection to a csv file.
        First row of the file includes the types of each column.

        :param file:
        :return:
        """
        self.db.to_csv(file)
        with open(file, "r") as original:
            content = original.readlines()
        content.insert(0, ','.join(self.types) + '\n')
        with open(file, "w") as modified:
            modified.writelines(content)

    def __len__(self):
        return len(self.db)

    def copy(self):
        """Return a copy of the object"""
        return self.from_db(self.db, like=self)

    @property
    def empty(self):
        return self.db.empty

    @property
    def types(self):
        return [typ if not isinstance(typ, dict) else next(iter(typ.keys())) for typ in self.lbl_types.values()]

    def categorical_values(self, cat_label: str):
        """
        Return available values (names) of specific Issue categorical label.
        """
        return list(self.categorical_table(cat_label).index)

    def prop_values(self, prop: str):
        """
        Return available values of specific Issue property
        """
        return list(self.db.index.get_level_values(prop).unique())

    def categorical_table(self, cat_label: str):
        """
        Return full table of specific Issue categorical label.
        """
        return next(iter(self.lbl_types[cat_label].values()))