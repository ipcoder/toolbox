from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class LabelRules:
    """Class encapsulates labeling rules in specific domain.
    The rules are loaded from domains labeling definition yaml file,
    which lists domain-specific labels categories and optionally
    the corresponding categorical values.

    Default domains definition is stored inside this package in 'labels.yml'
    with full path accessible as `LabelRules._default_domains_yaml`.

    Domain definition may include special _common_ domain (name in
    `LabelRules._common_domain`) with other domains inheriting from it.

    Available domains may be listed using _static_ `LabelRules.domains()` method.
    """
    _common_domain = 'common'

    def __init__(self, domain, path=None):
        """Create specific domain labeling rules from
        given or default domains definition file.

        :param domain: name of the domain to use
        :param path: path to a custom domains file
        """
        dms = self.load_domains(path)
        if domain not in dms:
            raise KeyError(f'Domain {domain} is not defined among {list(dms)}!')

        self.labels = getattr(dms, self._common_domain, {}).copy()  # type: dict
        self.labels.update(dms[domain])
        self.domain = domain
        for cat in self.labels:  # cast categorical as sets for fast searches
            self.labels[cat] = set(self.labels[cat])

    @staticmethod
    def domains(path=None):
        dms = LabelRules.load_domains(path)
        dms.pop(LabelRules._common_domain, {})
        return list(dms.keys())

    @staticmethod
    def load_domains(path=None):
        """Load domains conventions"""
        from toolbox.param import TBox
        from inu.env import EnvLoc
        # Consider: is that the right approach ? What if RES_LOC missing and path is None?
        path = path or (EnvLoc.RESOURCES / 'schemes ').first_file('labels.yml')
        dms = TBox.from_yaml(filename=path)
        return dms

    def __contains__(self, cat):
        return cat in self.labels

    def __getitem__(self, cat):
        if cat not in self:
            raise KeyError(f'Unknown category {cat} in the domain {self.domain}')
        return self.labels[cat]

    def in_cat(self, cat: str, item) -> bool:
        """Checks if item is defined in the given label category.
        :param cat: name of the category
        :param item: item to be found in this category

        :return True if label is categorical indicates if item
                is among the defined categories
                Otherwise returns None
        """
        items = self[cat]
        return item in items if items else True

    def __repr__(self):
        return f"Labeling Rules for {self.domain} domain\n" \
               f"Labels categories: {list(self.labels)}"
